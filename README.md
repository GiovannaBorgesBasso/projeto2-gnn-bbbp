# Projeto 2 — GNN para Predição de Permeabilidade da Barreira Hematoencefálica (BBBP)

Segundo projeto de uma série de projetos pessoais em cheminformatics/ML, desenvolvidos em paralelo ao meu trabalho no LabMol (UFG), com foco em construir um portfólio sólido para atuação como AI Engineer na indústria farmacêutica.

> **Projeto 1 (QSAR clássico):** [qsar-classico](https://github.com/GiovannaBorgesBasso/qsar-classico) — RDKit (ECFP4) + Random Forest/SVM para BTK (CHEMBL2842). AUC 0.95 (classificação), R² 0.74 (regressão).

---

## 🎯 Visão Geral

Neste projeto, a molécula deixa de ser representada como um **vetor fixo** (fingerprint) e passa a ser representada como **grafo nativo**: átomos são nós, ligações são arestas. Uma Graph Neural Network (GNN) aprende diretamente sobre essa estrutura, via *message passing*, em vez de depender de uma extração manual de features.

A tarefa é classificação binária: a molécula atravessa a barreira hematoencefálica (BBB) ou não — uma propriedade farmacocinética crítica para fármacos de ação central (ex: antidepressivos, antipsicóticos) e igualmente crítica de **evitar** em fármacos periféricos (ex: para minimizar neurotoxicidade).

---

## 🧠 GNN vs. Baseline Clássico

| | Projeto 1 (QSAR clássico) | Projeto 2 (GNN) |
|---|---|---|
| Representação | Fingerprint fixo (ECFP4, 2048 bits) | Grafo molecular (nós = átomos, arestas = ligações) |
| Extração de features | Manual, definida a priori | Aprendida pelo modelo durante o treino |
| Modelo | Random Forest / SVM | GCN / GAT (PyTorch Geometric) |
| Vantagem principal | Rápido, robusto em datasets pequenos | Captura topologia molecular completa |
| Risco principal | Perda de informação por colisão de hash | Precisa de mais dados para superar fingerprints |

---

## 📊 Dataset

- **MoleculeNet BBBP** — 2.039 moléculas, rótulo binário (BBB+/BBB-)
- **Desbalanceamento:** 76.5% BBB+ / 23.5% BBB-
- **Tamanho médio:** mediana de 23 átomos pesados, peso molecular mediano de 324 Da

---

## 📈 Resultados Finais

| Modelo | AUC-ROC (teste) |
|---|---|
| Random Forest + ECFP4 | 0.8420 |
| **SVM + ECFP4** | **0.8564** |
| GCN (grafos) | 0.8366 |
| GAT (grafos) | 0.8419 |

### Conclusão

Os modelos clássicos (especialmente SVM) superaram as GNNs neste dataset. Esse resultado é **esperado e documentado na literatura** para o BBBP por dois motivos:

1. **Dataset pequeno (~2.000 moléculas):** GNNs geralmente superam fingerprints a partir de ~10.000 moléculas. Com poucos dados, features cuidadosamente projetadas como ECFP4 são mais eficientes.

2. **Natureza da tarefa:** permeabilidade na BBB é fortemente correlacionada com propriedades globais simples (lipofilia, doadores/aceptores de H-bond, peso molecular) — exatamente o que ECFP4 captura bem.

A GAT (0.842) empatou com o Random Forest (0.842) e ficou apenas 1.4 pontos percentuais abaixo do SVM — não é uma derrota expressiva. GNNs são competitivas mesmo com poucos dados, mas não justificam a complexidade adicional nesse cenário específico. A vantagem das GNNs aparece com mais dados (ver Próximos Passos) ou com pré-treino em datasets maiores.

---

## 🏗️ Estrutura do Repositório

```
projeto2-gnn-bbbp/
├── data/
│   ├── raw/                  # dados originais baixados (não versionado)
│   └── processed/            # dados processados (não versionado)
├── notebooks/
│   └── eda_bbbp.py           # análise exploratória do dataset
├── src/
│   ├── data/
│   │   ├── mol_to_graph.py   # SMILES → objeto Data do PyG
│   │   └── dataset.py        # BBBPDataset (carrega 2039 moléculas)
│   ├── models/
│   │   ├── gcn.py            # arquitetura GCN
│   │   └── gat.py            # arquitetura GAT
│   ├── training/
│   │   ├── train.py          # loop de treino da GCN
│   │   └── train_gat.py      # loop de treino da GAT
│   ├── baseline/
│   │   └── classical.py      # RF + SVM + ECFP4 no mesmo split
│   └── utils/
│       ├── splits.py         # scaffold split estratificado
│       └── plot_results.py   # visualizações finais
├── models/checkpoints/        # pesos salvos (não versionado)
├── results/
│   ├── figures/              # gráficos de comparação e curvas de treino
│   └── metrics/              # métricas em JSON (baseline, GCN, GAT)
├── reports/
│   └── relatorio_final.md    # relatório completo com análise e conclusões
├── environment.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup do Ambiente

```bash
conda create -n gnn-bbbp python=3.11 -y
conda activate gnn-bbbp
pip install rdkit torch torch_geometric deepchem matplotlib seaborn scikit-learn
```

---

## 🚀 Como Reproduzir

```bash
# 1. EDA
python notebooks/eda_bbbp.py

# 2. Baseline clássico
python src/baseline/classical.py

# 3. Treinar GCN
python src/training/train.py

# 4. Treinar GAT
python src/training/train_gat.py

# 5. Gerar visualizações finais
python src/utils/plot_results.py
```

---

## 🗺️ Roadmap

- [x] **Etapa 0** — Setup do ambiente e estrutura do repositório
- [x] **Etapa 1** — Conceitos: grafos moleculares, message passing, objetos `Data` do PyG
- [x] **Etapa 2** — EDA do dataset BBBP (distribuição de classes, tamanho molecular)
- [x] **Etapa 3** — Pipeline SMILES → grafo (`mol_to_graph.py`, `BBBPDataset`)
- [x] **Etapa 4** — Scaffold split estratificado + `DataLoader`
- [x] **Etapa 5** — Baseline clássico (ECFP4 + RF/SVM) no mesmo split
- [x] **Etapa 6** — GCN implementada e treinada (AUC teste: 0.8366)
- [x] **Etapa 7** — GAT implementada e treinada (AUC teste: 0.8419)
- [x] **Etapa 8** — Comparação final, relatório e visualizações
- [ ] **Etapa 9** — (Próximo passo) Extensão para B3DB (~7.807 moléculas)

---

## 🔬 Decisões Metodológicas

**Por que scaffold split e não random split?**
Moléculas estruturalmente parecidas tendem a ter o mesmo rótulo. Um split aleatório permite que variantes do mesmo scaffold caiam em treino e teste simultaneamente, inflando a performance artificialmente. O scaffold split garante que scaffolds inteiros fiquem em apenas um conjunto.

**Por que scaffold split estratificado?**
O scaffold split padrão do BBBP (ordenar grupos por tamanho, preencher treino primeiro) coloca todos os compostos BBB- no treino, deixando validação e teste com 0 compostos BBB- — tornando AUC-ROC indefinida. A versão estratificada distribui scaffolds por classe majoritária, garantindo ambas as classes em todos os conjuntos.

**Por que AUC-ROC e não acurácia?**
Com 76.5% BBB+, um modelo que sempre responde "BBB+" acerta 76% das vezes sem aprender nada. AUC-ROC mede a capacidade de separar as classes independentemente do desbalanceamento.

---

## 🔗 Próximos Passos

- **B3DB (~7.807 moléculas):** replicar o pipeline num dataset maior para verificar se GNNs superam fingerprints com mais dados
- **Pré-treino:** fine-tuning de modelos GNN pré-treinados em milhões de moléculas
- **Otimização de hiperparâmetros:** grid search sobre hidden_channels, num_layers, dropout

---

## 📚 Referências

- Martins et al. (2012). *A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling.*
- Wu et al. (2018). *MoleculeNet: A Benchmark for Molecular Machine Learning.*
- Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.*
- Veličković et al. (2018). *Graph Attention Networks.*
- Hu et al. (2020). *Strategies for Pre-Training Graph Neural Networks.*

## 🔗 Projetos Relacionados

- [Projeto 1 — QSAR Clássico (BTK)](https://github.com/GiovannaBorgesBasso/qsar-classico)
