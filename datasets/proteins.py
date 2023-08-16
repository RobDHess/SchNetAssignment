import torch
from torch_geometric.data import InMemoryDataset, Data
import gdown
import prody as pd
import os
import numpy as np

amino_acid_conversion = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "CME": "C",  # Mapping CME to Cysteine's single-letter code
}


class ProteinDataset(InMemoryDataset):
    """A dataset of protein structures."""

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        llm_transform=None,
    ):
        self.url = (
            "https://drive.google.com/drive/folders/1bbXrThtiH1jQBvvbq_G8Xd3CEVSIT9Cx"
        )
        self.llm_transform = llm_transform

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw", "data")

    @property
    def processed_dir(self):
        if self.llm_transform is None:
            return os.path.join(self.root, "processed")
        return os.path.join(self.root, "processed", self.llm_transform.name)

    @property
    def raw_file_names(self):
        return [
            "1a05_1_protein.pdb",
            "1a1b_1_protein.pdb",
            "1a2c_1_protein.pdb",
            "1a4g_1_protein.pdb",
            "1a4m_3_protein.pdb",
            "1a5u_1_protein.pdb",
            "1a0f_1_protein.pdb",
            "1a1c_1_protein.pdb",
            "1a2t_1_protein.pdb",
            "1a4h_1_protein.pdb",
            "1a4m_4_protein.pdb",
            "1a5u_2_protein.pdb",
            "1a0g_1_protein.pdb",
            "1a1e_1_protein.pdb",
            "1a2u_1_protein.pdb",
            "1a4i_1_protein.pdb",
            "1a4q_1_protein.pdb",
            "1a5v_1_protein.pdb",
            "1a0j_1_protein.pdb",
            "1a1m_1_protein.pdb",
            "1a3k_1_protein.pdb",
            "1a4k_1_protein.pdb",
            "1a4r_1_protein.pdb",
            "1a5w_1_protein.pdb",
            "1a0q_1_protein.pdb",
            "1a1n_1_protein.pdb",
            "1a3t_1_protein.pdb",
            "1a4k_2_protein.pdb",
            "1a4w_1_protein.pdb",
            "1a5x_1_protein.pdb",
            "1a0t_1_protein.pdb",
            "1a1o_1_protein.pdb",
            "1a3u_1_protein.pdb",
            "1a4m_1_protein.pdb",
            "1a5b_1_protein.pdb",
            "1a5z_1_protein.pdb",
            "1a1a_1_protein.pdb",
            "1a2b_1_protein.pdb",
            "1a3v_1_protein.pdb",
            "1a4m_2_protein.pdb",
            "1a5s_1_protein.pdb",
            "1a6v_1_protein.pdb",
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        gdown.download_folder(url=self.url, output=self.raw_dir)

    def process(self):
        """Converts the raw data into torch-geometric's internal data format."""
        # Read data into huge `Data` list.
        data_list = []

        labels = {}
        label_path = os.path.join(self.root, "raw", "labels.txt")
        with open(label_path, "r") as file:
            for line in file:
                protein, value = line.strip().split()
                labels[protein] = float(value)

        for filename in self.raw_file_names:
            path = os.path.join(self.raw_dir, filename)
            name = filename.split(".")[0]
            amino_acids, positions = extract_calpha_positions(path)

            sequence = "".join([amino_acid_conversion[aa] for aa in amino_acids])
            positions = np.array(positions)
            assert len(sequence) == len(positions)

            data_list.append(
                Data(
                    name=name,
                    sequence=sequence,
                    pos=torch.from_numpy(positions).float(),
                    y=torch.FloatTensor([labels[name]]),
                )
            )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if self.llm_transform is not None:
            data_list = [self.llm_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def extract_calpha_positions(pdb_filename):
    """Extracts the Cα positions from a PDB file."""
    # Load the protein structure
    structure = pd.parsePDB(pdb_filename)

    # Extract Cα atoms
    calphas = structure.select("name CA")

    amino_acids = []
    positions = []

    for ca in calphas:
        amino_acids.append(ca.getResname())
        positions.append(ca.getCoords())

    return amino_acids, positions
