"""
src/utils/splits.py

Scaffold split estratificado para o dataset BBBP.

Por que estratificado:
    O scaffold split padrão (ordenar por tamanho de grupo) coloca todos os
    compostos BBB- no treino no dataset BBBP, porque as moléculas BBB- têm
    scaffolds únicos (singletons) e o algoritmo preenche o treino primeiro
    com os grandes grupos BBB+. Resultado: val e teste ficam com 0 BBB-,
    tornando AUC-ROC indefinida.

    A versão estratificada separa os scaffolds por classe majoritária e
    distribui cada grupo proporcionalmente. Mantém a filosofia do scaffold
    split (nenhum scaffold aparece em dois conjuntos) e garante as duas
    classes em todos os conjuntos.

Uso:
    from src.data.dataset import BBBPDataset
    from src.utils.splits import scaffold_split

    dataset = BBBPDataset()
    train, val, test = scaffold_split(dataset, dataset.smiles_list, dataset.labels)
"""

from collections import defaultdict
import numpy as np
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


def _split_groups(groups, train_ratio, val_ratio):
    """Distribui grupos de índices em treino/val/teste."""
    all_idx = [i for g in groups for i in g]
    n = len(all_idx)
    train_cut = int(train_ratio * n)
    val_cut   = int((train_ratio + val_ratio) * n)

    train_idx, val_idx, test_idx = [], [], []
    count = 0
    for g in groups:
        if count + len(g) <= train_cut:
            train_idx.extend(g)
        elif count + len(g) <= val_cut:
            val_idx.extend(g)
        else:
            test_idx.extend(g)
        count += len(g)
    return train_idx, val_idx, test_idx


def scaffold_split(
    dataset,
    smiles_list: list[str],
    labels: list[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Divide um dataset em treino/validação/teste por scaffold (estratificado).

    Algoritmo:
        1. Calcula o scaffold de cada molécula
        2. Agrupa moléculas pelo scaffold
        3. Separa grupos por classe majoritária (BBB+ vs BBB-)
        4. Embaralha cada grupo com seed fixo
        5. Distribui proporcionalmente, garantindo ambas as classes em val/teste

    Args:
        dataset:     lista de objetos Data do PyG (BBBPDataset).
        smiles_list: lista de SMILES na mesma ordem que o dataset.
        labels:      lista de rótulos (0 ou 1) na mesma ordem que o dataset.
        train_ratio: fração para treino (padrão 0.8).
        val_ratio:   fração para validação (padrão 0.1).
        test_ratio:  fração para teste (padrão 0.1).
        seed:        semente para reproducibilidade.

    Returns:
        Tupla (train_dataset, val_dataset, test_dataset) como listas.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio deve ser 1.0"

    y = np.array(labels)

    # 1. Agrupar índices por scaffold
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        if scaffold is not None:
            scaffolds[scaffold].append(idx)

    # 2. Separar scaffolds por classe majoritária
    pos_scaffolds = []
    neg_scaffolds = []
    for indices in scaffolds.values():
        if y[indices].mean() >= 0.5:
            pos_scaffolds.append(indices)
        else:
            neg_scaffolds.append(indices)

    # 3. Embaralhar com seed fixo
    rng = np.random.default_rng(seed)
    rng.shuffle(pos_scaffolds)
    rng.shuffle(neg_scaffolds)

    # 4. Distribuir cada grupo proporcionalmente
    pos_train, pos_val, pos_test = _split_groups(pos_scaffolds, train_ratio, val_ratio)
    neg_train, neg_val, neg_test = _split_groups(neg_scaffolds, train_ratio, val_ratio)

    train_idx = pos_train + neg_train
    val_idx   = pos_val   + neg_val
    test_idx  = pos_test  + neg_test

    # 5. Montar subsets
    train_set = [dataset[i] for i in train_idx]
    val_set   = [dataset[i] for i in val_idx]
    test_set  = [dataset[i] for i in test_idx]

    n = len(dataset)
    print("Scaffold split estratificado concluído:")
    print(f"  Treino:    {len(train_set)} moléculas ({len(train_set)/n*100:.1f}%) "
          f"| BBB+={y[train_idx].sum()} BBB-={(y[train_idx]==0).sum()}")
    print(f"  Validação: {len(val_set)} moléculas ({len(val_set)/n*100:.1f}%) "
          f"| BBB+={y[val_idx].sum()} BBB-={(y[val_idx]==0).sum()}")
    print(f"  Teste:     {len(test_set)} moléculas ({len(test_set)/n*100:.1f}%) "
          f"| BBB+={y[test_idx].sum()} BBB-={(y[test_idx]==0).sum()}")

    return train_set, val_set, test_set
