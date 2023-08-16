import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


class ProteinRegressionModel(pl.LightningModule):
    """Regress property for protein structures"""

    def __init__(
        self,
        model,
        lr,
    ):
        super().__init__()
        self.model = model
        self.lr = lr

        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.test_metric = torchmetrics.MeanAbsoluteError()

    def forward(self, graph):
        pred = self.model(graph).squeeze(-1)
        return pred

    def training_step(self, graph):
        pred = self(graph)
        loss = F.mse_loss(pred, graph.y)
        self.train_metric(pred, graph.y)
        return loss

    def on_train_epoch_end(self):
        self.log("Train MAE", self.train_metric, prog_bar=True)

    def validation_step(self, graph, batch_idx):
        pred = self(graph)
        self.valid_metric(pred, graph.y)

    def on_validation_epoch_end(self):
        self.log("Val MAE", self.valid_metric, prog_bar=True)

    def test_step(self, graph, batch_idx):
        pred = self(graph)
        self.test_metric(pred, graph.y)

    def on_test_epoch_end(self):
        self.log("Test MAE", self.test_metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.estimated_stepping_batches
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]
