"""
src/utils/plot_results.py

Gera as visualizações finais do projeto:
    1. Comparação de AUC-ROC entre todos os modelos
    2. Curvas de treino (val AUC por época) da GCN e GAT

Como rodar (da raiz do projeto):
    python src/utils/plot_results.py

Saída:
    results/figures/comparacao_modelos.png
    results/figures/curvas_treino.png
"""

import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_comparacao(output_path):
    """Gráfico de barras comparando AUC-ROC de todos os modelos."""

    # carregar métricas do baseline
    baseline = load_json("results/metrics/baseline.json")
    rf_auc   = next(m['auc_roc'] for m in baseline['baseline'] if 'Random' in m['modelo'])
    svm_auc  = next(m['auc_roc'] for m in baseline['baseline'] if 'SVM' in m['modelo'])

    # carregar métricas das GNNs
    gcn_auc = load_json("results/metrics/gcn_training.json")['test_auc']
    gat_auc = load_json("results/metrics/gat_training.json")['test_auc']

    modelos = ['RF + ECFP4\n(baseline)', 'SVM + ECFP4\n(baseline)',
               'GCN\n(grafos)', 'GAT\n(grafos)']
    aucs  = [rf_auc, svm_auc, gcn_auc, gat_auc]
    cores = ['#aec6e8', '#aec6e8', '#f4a582', '#d6604d']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(modelos, aucs, color=cores, edgecolor='white', linewidth=1.5, width=0.5)

    ax.set_ylim(max(0, min(aucs) - 0.05), min(1.0, max(aucs) + 0.05))
    ax.set_ylabel('AUC-ROC (conjunto de teste)', fontsize=12)
    ax.set_title('Comparação de Modelos — BBBP (Scaffold Split Estratificado)',
                 fontsize=12, fontweight='bold', pad=15)
    ax.axhline(svm_auc, color='steelblue', linestyle='--',
               linewidth=1.2, alpha=0.6, label=f'Melhor baseline (SVM): {svm_auc:.4f}')

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

    ax.legend(fontsize=10)

    # legenda de cores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#aec6e8', label='Baseline clássico (ECFP4)'),
        Patch(facecolor='#f4a582', label='GCN'),
        Patch(facecolor='#d6604d', label='GAT'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {output_path}")


def plot_curvas_treino(output_path):
    """Curvas de val AUC por época da GCN e GAT."""

    gcn_data = load_json("results/metrics/gcn_training.json")['historico']
    gat_data = load_json("results/metrics/gat_training.json")['historico']

    gcn_epochs = [d['epoch'] for d in gcn_data]
    gcn_aucs   = [d['val_auc'] for d in gcn_data]
    gat_epochs = [d['epoch'] for d in gat_data]
    gat_aucs   = [d['val_auc'] for d in gat_data]

    gcn_best = max(gcn_aucs)
    gat_best = max(gat_aucs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gcn_epochs, gcn_aucs, color='#f4a582', linewidth=2,
            label=f'GCN (hidden=64) — melhor val AUC: {gcn_best:.4f}')
    ax.plot(gat_epochs, gat_aucs, color='#d6604d', linewidth=2,
            label=f'GAT (hidden=128) — melhor val AUC: {gat_best:.4f}')

    ax.set_xlabel('Época', fontsize=11)
    ax.set_ylabel('AUC-ROC (validação)', fontsize=11)
    ax.set_title('Curvas de Treino — Val AUC por Época',
                 fontsize=12, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {output_path}")


if __name__ == "__main__":
    plot_comparacao(os.path.join(OUTPUT_DIR, "comparacao_modelos.png"))
    plot_curvas_treino(os.path.join(OUTPUT_DIR, "curvas_treino.png"))
    print("\nVisualizações geradas em results/figures/")
