"""
src/data/dataset.py

Carrega o dataset BBBP e converte todas as moléculas em grafos PyG.

Uso:
    from src.data.dataset import BBBPDataset
    dataset = BBBPDataset()
    print(len(dataset))       # 2039
    print(dataset[0])         # Data(x=[20, 6], edge_index=[2, 40], ...)
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.data.mol_to_graph import smiles_to_graph

BBBP_URL = (
    "https://raw.githubusercontent.com/GLambard/"
    "Molecules_Dataset_Collection/master/latest/BBBP.csv"
)


class BBBPDataset:
    """
    Dataset BBBP (MoleculeNet) como lista de objetos Data do PyG.

    Cada item é uma molécula representada como grafo:
        data.x          — features dos átomos, shape (num_atoms, 6)
        data.edge_index — conectividade, shape (2, num_edges)
        data.edge_attr  — features das ligações, shape (num_edges, 3)
        data.y          — rótulo binário: 1=BBB+, 0=BBB-

    Args:
        url: URL ou caminho local do CSV do BBBP.
             Por padrão baixa do repositório público do MoleculeNet.
    """

    def __init__(self, url: str = BBBP_URL):
        print("Carregando dataset BBBP...")
        df = pd.read_csv(url)
        df = df[['smiles', 'p_np']].rename(columns={'p_np': 'label'})
        df = df.dropna(subset=['smiles'])
        df['smiles'] = df['smiles'].astype(str)

        self.data_list = []
        invalidos = 0

        for _, row in df.iterrows():
            graph = smiles_to_graph(row['smiles'], label=int(row['label']))
            if graph is not None:
                self.data_list.append(graph)
            else:
                invalidos += 1

        print(f"Moléculas carregadas: {len(self.data_list)}")
        if invalidos:
            print(f"SMILES inválidos ignorados: {invalidos}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        return self.data_list[idx]
