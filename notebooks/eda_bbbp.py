"""
Etapa 2 — EDA do dataset BBBP.

O que vamos descobrir aqui:
- Quantas moléculas tem o dataset e se está balanceado
- Tamanho médio das moléculas (afeta o custo computacional da GNN)
- Distribuição de peso molecular (referência para regra de Lipinski/CNS-MPO)

Como rodar:
    python eda_bbbp.py

Requer: pandas, rdkit, matplotlib, seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings("ignore")

# ── 1. Carregar o dataset ────────────────────────────────────────────────────

URL = "https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest/BBBP.csv"

print("Baixando dataset BBBP...")
df = pd.read_csv(URL)
df = df[['name', 'smiles', 'p_np']].rename(columns={'p_np': 'label'})
df = df.dropna(subset=['smiles'])
df['smiles'] = df['smiles'].astype(str)
print(f"Dataset carregado: {len(df)} moléculas\n")

# ── 2. Calcular descritores básicos ─────────────────────────────────────────

def get_num_atoms(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumAtoms() if mol else None

def get_mol_weight(smi):
    mol = Chem.MolFromSmiles(smi)
    return round(Descriptors.MolWt(mol), 2) if mol else None

print("Calculando descritores moleculares...")
df['num_atoms'] = df['smiles'].apply(get_num_atoms)
df['mol_weight'] = df['smiles'].apply(get_mol_weight)

invalidos = df['num_atoms'].isna().sum()
df = df.dropna(subset=['num_atoms'])
print(f"SMILES inválidos removidos: {invalidos}")
print(f"Moléculas válidas para análise: {len(df)}\n")

# ── 3. Resumo do dataset ─────────────────────────────────────────────────────

bbb_pos = (df['label'] == 1).sum()
bbb_neg = (df['label'] == 0).sum()

print("=" * 45)
print("RESUMO DO DATASET BBBP")
print("=" * 45)
print(f"Total de moléculas:     {len(df)}")
print(f"BBB+ (atravessa):       {bbb_pos} ({bbb_pos/len(df)*100:.1f}%)")
print(f"BBB- (não atravessa):   {bbb_neg} ({bbb_neg/len(df)*100:.1f}%)")
print()
print(f"Átomos pesados por molécula:")
print(f"  mínimo:  {int(df['num_atoms'].min())}")
print(f"  máximo:  {int(df['num_atoms'].max())}")
print(f"  mediana: {int(df['num_atoms'].median())}")
print(f"  média:   {df['num_atoms'].mean():.1f}")
print()
print(f"Peso molecular (Da):")
print(f"  mínimo:  {df['mol_weight'].min():.1f}")
print(f"  máximo:  {df['mol_weight'].max():.1f}")
print(f"  mediana: {df['mol_weight'].median():.1f}")
print(f"  média:   {df['mol_weight'].mean():.1f}")
print("=" * 45)

# ── 4. Gráficos ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("EDA — Dataset BBBP (MoleculeNet)", fontsize=14, fontweight='bold')

# --- Gráfico 1: Distribuição de classes ---
ax = axes[0]
cores = ['#e07070', '#70a0e0']  # vermelho suave = BBB-, azul suave = BBB+
contagens = [bbb_neg, bbb_pos]
rotulos = [f'BBB-\n(não atravessa)\n{bbb_neg} ({bbb_neg/len(df)*100:.1f}%)',
           f'BBB+\n(atravessa)\n{bbb_pos} ({bbb_pos/len(df)*100:.1f}%)']
bars = ax.bar(rotulos, contagens, color=cores, edgecolor='white', linewidth=1.5)
ax.set_title('Distribuição de Classes', fontweight='bold')
ax.set_ylabel('Número de moléculas')
ax.set_ylim(0, max(contagens) * 1.15)
for bar, val in zip(bars, contagens):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(val), ha='center', fontweight='bold')

# --- Gráfico 2: Distribuição de número de átomos ---
ax = axes[1]
ax.hist(df[df['label'] == 1]['num_atoms'], bins=40, alpha=0.6,
        color='#70a0e0', label='BBB+', edgecolor='white')
ax.hist(df[df['label'] == 0]['num_atoms'], bins=40, alpha=0.6,
        color='#e07070', label='BBB-', edgecolor='white')
ax.axvline(df['num_atoms'].median(), color='black', linestyle='--',
           linewidth=1.2, label=f'mediana = {int(df["num_atoms"].median())}')
ax.set_title('Número de Átomos Pesados', fontweight='bold')
ax.set_xlabel('Número de átomos')
ax.set_ylabel('Frequência')
ax.legend()

# --- Gráfico 3: Distribuição de peso molecular ---
ax = axes[2]
ax.hist(df[df['label'] == 1]['mol_weight'], bins=40, alpha=0.6,
        color='#70a0e0', label='BBB+', edgecolor='white')
ax.hist(df[df['label'] == 0]['mol_weight'], bins=40, alpha=0.6,
        color='#e07070', label='BBB-', edgecolor='white')
ax.axvline(500, color='orange', linestyle='--', linewidth=1.2,
           label='Lipinski MW ≤ 500')
ax.axvline(df['mol_weight'].median(), color='black', linestyle='--',
           linewidth=1.2, label=f'mediana = {int(df["mol_weight"].median())} Da')
ax.set_title('Peso Molecular', fontweight='bold')
ax.set_xlabel('Peso molecular (Da)')
ax.set_ylabel('Frequência')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('eda_bbbp.png', dpi=150, bbox_inches='tight')
print("\nGráfico salvo: eda_bbbp.png")
plt.show()