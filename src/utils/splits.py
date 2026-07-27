"""
src/utils/splits.py

Scaffold split para o dataset BBBP.

O que é scaffold split e por que usamos:
    Scaffold é o esqueleto central de uma molécula (núcleo aromático/cíclico
    sem as cadeias laterais), extraído pelo algoritmo de Bemis-Murcko.
    Moléculas com o mesmo scaffold são estruturalmente parecidas.

    No split aleatório, moléculas parecidas podem cair em treino E teste,
    fazendo o modelo "acertar" no teste por ter visto estruturas similares
    no treino — inflando a performance artificialmente.

    No scaffold split, cada scaffold fica inteiro num único conjunto.
    O modelo é forçado a generalizar para estruturas que nunca viu,
    simulando o uso real em drug discovery.

Uso:
    from src.data.dataset import BBBPDataset
    from src.utils.splits import scaffold_split

    dataset = BBBPDataset()
    train, val, test = scaffold_split(dataset)
"""

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
warnings.filterwarnings("ignore")


def get_scaffold(smiles: str) -> str | None:
    """
    Extrai o scaffold de Bemis-Murcko de um SMILES.

    Args:
        smiles: string SMILES da molécula.

    Returns:
        SMILES do scaffold, ou None se o SMILES for inválido.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(
    dataset,
    smiles_list: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple:
    """
    Divide um dataset em treino/validação/teste por scaffold.

    Algoritmo (padrão DeepChem):
        1. Calcula o scaffold de cada molécula
        2. Agrupa moléculas pelo scaffold
        3. Ordena os grupos do maior para o menor
        4. Distribui os grupos nos conjuntos sem dividir nenhum grupo

    Args:
        dataset:     lista de objetos Data do PyG (BBBPDataset).
        smiles_list: lista de SMILES na mesma ordem que o dataset.
        train_ratio: fração para treino (padrão 0.8).
        val_ratio:   fração para validação (padrão 0.1).
        test_ratio:  fração para teste (padrão 0.1).

    Returns:
        Tupla (train_dataset, val_dataset, test_dataset) como listas.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio deve ser 1.0"

    n = len(dataset)
    train_size = int(train_ratio * n)
    val_size   = int(val_ratio * n)

    # 1. Agrupar índices por scaffold
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        if scaffold is not None:
            scaffolds[scaffold].append(idx)

    # 2. Ordenar do maior grupo para o menor
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    # 3. Distribuir grupos
    train_idx, val_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= train_size:
            train_idx.extend(scaffold_set)
        elif len(val_idx) + len(scaffold_set) <= val_size:
            val_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    # 4. Montar subsets
    train_set = [dataset[i] for i in train_idx]
    val_set   = [dataset[i] for i in val_idx]
    test_set  = [dataset[i] for i in test_idx]

    print(f"Scaffold split concluído:")
    print(f"  Treino:    {len(train_set)} moléculas ({len(train_set)/n*100:.1f}%)")
    print(f"  Validação: {len(val_set)} moléculas ({len(val_set)/n*100:.1f}%)")
    print(f"  Teste:     {len(test_set)} moléculas ({len(test_set)/n*100:.1f}%)")

    return train_set, val_set, test_set
