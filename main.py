import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric as tg
import warnings

from datasets.proteins import ProteinDataset
from nn.schnet import SchNet
from nn.ssp import ShiftedSoftplus
from tasks.protein_regression import ProteinRegressionModel
import utils.transforms as T


parser = argparse.ArgumentParser()

# Data settings
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--log", type=bool, default=False)
parser.add_argument("--gpus", type=int, default=-1)
parser.add_argument("--num_workers", type=int, default=-1)
parser.add_argument("--dataset", type=str, default="Proteins")

# Train settings
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--data_seed", type=int, default=None)
parser.add_argument("--test", type=bool, default=False)

# Model settings
parser.add_argument("--model", type=str, default="SchNet")
parser.add_argument("--in_features", type=int, default=320)
parser.add_argument("--hidden_features", type=int, default=64)
parser.add_argument("--out_features", type=int, default=1)
parser.add_argument("--depth", type=int, default=3)
parser.add_argument("--aggr", type=str, default="add")
parser.add_argument("--pool", type=str, default="add")
parser.add_argument("--cutoff", type=float, default=30)
parser.add_argument("--act", type=str, default="ssp")


# Weight net settings
parser.add_argument("--weight_net_dims", type=int, nargs="+", default=[64])
parser.add_argument("--num_basis", type=int, default=300)
parser.add_argument("--d_min", type=float, default=0)
parser.add_argument("--d_max", type=float, default=30)
parser.add_argument("--gamma", type=float, default=10)

# ESM settings
parser.add_argument("--esm_name", type=str, default="esm2_t6_8M_UR50D")
parser.add_argument("--esm_device", type=str, default="cuda:0")


args = parser.parse_args()

act_dict = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}

if __name__ == "__main__":
    # Checks
    assert args.d_min < args.d_max, "Please set d_min < d_max"
    assert args.pool in ["add", "mean", "max"]
    assert args.aggr in ["add", "mean", "max"]
    assert args.act in act_dict.keys()

    if args.cutoff != args.d_max:
        warning = "Cutoff={} and d_max={} are not equal, this may lead to suboptimal results!".format(
            args.cutoff, args.d_max
        )
        warnings.warn(warning)

    # Reproducibility
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
        deterministic = True
    else:
        deterministic = False

    # Devices
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count()
    if args.num_workers == -1:
        args.num_workers = int(os.cpu_count() / 4)

    # Dataset
    if args.dataset == "Proteins":
        transform = tg.transforms.Compose(
            [
                tg.transforms.RadiusGraph(args.cutoff),
                tg.transforms.Distance(),
            ]
        )

        esm_transform = T.ESMTransform(args.esm_name, args.esm_device)

        dataset = ProteinDataset(
            root=args.root, llm_transform=esm_transform, transform=transform
        )
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(args.data_seed)
            if args.data_seed
            else None,
        )

    # Print dataset sizes
    print("Train dataset size: {}".format(len(train_dataset)))
    print("Valid dataset size: {}".format(len(valid_dataset)))
    print("Test dataset size: {}".format(len(test_dataset)))

    dataloaders = [
        tg.loader.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=dataset == train_dataset,
        )
        for dataset in [train_dataset, valid_dataset, test_dataset]
    ]

    # Model
    if args.model == "SchNet":
        model = SchNet(
            in_features=args.in_features,
            hidden_features=args.hidden_features,
            out_features=args.out_features,
            depth=args.depth,
            weight_net_dims=args.weight_net_dims,
            num_basis=args.num_basis,
            d_min=args.d_min,
            d_max=args.d_max,
            gamma=args.gamma,
            act=act_dict[args.act],
            aggr=args.aggr,
            pooler=args.pool,
        )
    else:
        raise ValueError("Model {} not implemented".format(args.model))

    # Task
    model = ProteinRegressionModel(model, lr=args.lr)

    # Logging
    if args.log:
        logger = pl.loggers.WandbLogger(
            project=" ".join(["ProteinRegression", args.dataset]),
            name=args.model,
            config=args,
        )
    else:
        logger = None

    # Let's go!
    print(model)
    print(args)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        deterministic=deterministic,
    )

    trainer.fit(model, dataloaders[0], dataloaders[1])
    if args.test:
        trainer.test(dataloaders=dataloaders[2], ckpt_path="best")
