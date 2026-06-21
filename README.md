# Projeto 2 — GNN para Predição de Permeabilidade da Barreira Hematoencefálica (BBBP)

Segundo projeto de uma série de projetos pessoais em cheminformatics/ML, desenvolvidos em paralelo ao meu trabalho no LabMol (UFG), com foco em construir um portfólio sólido para atuação como AI Engineer na indústria farmacêutica.

> **Projeto 1 (QSAR clássico):** [qsar-classico](https://github.com/GiovannaBorgesBasso/qsar-classico) — RDKit (ECFP4) + Random Forest/SVM para BTK (CHEMBL2842). AUC 0.95 (classificação), R² 0.74 (regressão).

## 🎯 Visão Geral

Neste projeto, a molécula deixa de ser representada como um **vetor fixo** (fingerprint) e passa a ser representada como **grafo nativo**: átomos são nós, ligações são arestas. Uma Graph Neural Network (GNN) aprende diretamente sobre essa estrutura, via *message passing*, em vez de depender de uma extração manual de features.

A tarefa é classificação binária: a molécula atravessa a barreira hematoencefálica (BBB) ou não — uma propriedade farmacocinética crítica para fármacos de ação central (ex: antidepressivos, antipsicóticos) e igualmente crítica de **evitar** em fármacos periféricos (ex: para minimizar neurotoxicidade).

## 🧠 Por que GNN, e por que comparar com o Projeto 1?

| | Projeto 1 (QSAR clássico) | Projeto 2 (GNN) |
|---|---|---|
| Representação | Fingerprint fixo (ECFP4, 2048 bits) | Grafo molecular (nós = átomos, arestas = ligações) |
| Extração de features | Manual, definida a priori (algoritmo de hashing) | Aprendida pelo modelo durante o treino |
| Modelo | Random Forest / SVM | GCN / GAT (PyTorch Geometric) |
| Vantagem principal | Rápido, interpretável, robusto em datasets pequenos | Captura topologia molecular completa, sem perda por colisão de hash |
| Risco principal | Perda de informação estrutural (colisões de hash, raio fixo) | Precisa de mais dados / regularização para não overfit |

O objetivo central deste projeto **não é "provar que GNN é melhor"** — é fazer a comparação honesta, no mesmo dataset e no mesmo split, e entender em que condições cada abordagem se sai melhor. Essa é uma pergunta real do dia a dia de um time de drug discovery.

## 📊 Dataset

- **Principal:** [MoleculeNet BBBP](https://moleculenet.org/) — ~2.050 moléculas, rótulo binário (BBB+/BBB-).
- **Extensão opcional (fase posterior):** [B3DB](https://github.com/theochem/B3DB) — ~7.807 moléculas, dataset mais recente e mais balanceado, útil para testar se as conclusões se mantêm em escala maior.

> **Nota metodológica importante:** o BBBP é conhecido por ter um *scaffold split* mais informativo que um split aleatório (moléculas estruturalmente relacionadas tendem a ter o mesmo rótulo, então um split aleatório infla a performance). Vamos usar **scaffold split** como split principal, e reportar random split apenas como referência/diagnóstico — isso vale tanto para o baseline clássico re-treinado quanto para a GNN, para a comparação ser justa.

## 🏗️ Estrutura do Repositório

```
projeto2-gnn-bbbp/
├── data/
│   ├── raw/                  # dados originais baixados (não versionado)
│   └── processed/            # dados processados / objetos PyG (não versionado)
├── notebooks/                # exploração, uma etapa por notebook
├── src/
│   ├── data/                 # download, parsing SMILES, conversão mol→grafo
│   ├── models/                # arquiteturas GCN, GAT
│   ├── training/              # loops de treino, avaliação, métricas
│   ├── baseline/               # baseline clássico (fingerprint + RF/SVM) no mesmo split
│   └── utils/                  # splits, seeds, helpers gerais
├── models/checkpoints/        # pesos salvos (não versionado)
├── results/
│   ├── figures/                # gráficos finais
│   └── metrics/                # tabelas de métricas (json/csv)
├── reports/                    # relatório final de comparação
├── tests/                      # testes unitários (ex: conversão SMILES→grafo)
├── environment.yml
├── requirements.txt
└── README.md
```

`data/`, `models/checkpoints/` e artefatos grandes ficam fora do versionamento (ver `.gitignore`) — só o código e os notebooks (com outputs limpos ou leves) vão para o Git.

## ⚙️ Setup do Ambiente

PyTorch Geometric tem instalação sensível a hardware (CPU vs. GPU/CUDA), então o setup é em duas etapas: stack científica geral via conda, depois PyTorch/PyG via pip ajustado à sua máquina.

### 1. Criar o ambiente base

```bash
conda env create -f environment.yml
conda activate gnn-bbbp
```

### 2. Instalar PyTorch (escolha conforme seu hardware)

```bash
# CPU apenas
pip install torch --index-url https://download.pytorch.org/whl/cpu

# OU GPU com CUDA 12.4 (exemplo — confira sua versão de CUDA com `nvidia-smi`)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Instalar PyTorch Geometric

Desde a versão 2.3, o PyG funciona com instalação mínima (sem extensões compiladas):

```bash
pip install torch_geometric
```

### 4. Verificar a instalação

```bash
python -c "import torch, torch_geometric; print('torch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA disponível:', torch.cuda.is_available())"
```

> Se aparecer algum erro de versão incompatível entre torch e PyG, confira a [tabela oficial de compatibilidade](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) antes de tentar forçar instalação.

## 🗺️ Roadmap

- [x] **Etapa 0** — Setup do ambiente e estrutura do repositório
- [ ] **Etapa 1** — Conceitos: grafos moleculares, message passing, `Data` objects do PyG
- [ ] **Etapa 2** — EDA do dataset BBBP (distribuição de classes, tamanho molecular, scaffolds)
- [ ] **Etapa 3** — Pipeline SMILES → grafo (features de nó/aresta, construção do `Dataset` PyG)
- [ ] **Etapa 4** — Scaffold split + `DataLoader`, definição do protocolo de avaliação
- [ ] **Etapa 5** — Baseline clássico (ECFP4 + RF/SVM) re-treinado no mesmo split, para comparação justa
- [ ] **Etapa 6** — Implementação e treino de uma GCN
- [ ] **Etapa 7** — Implementação e treino de uma GAT
- [ ] **Etapa 8** — Comparação final (GCN vs. GAT vs. baseline clássico) + análise de erros
- [ ] **Etapa 9** — (Opcional) Extensão para B3DB, validação de robustez em escala maior
- [ ] **Etapa 10** — Documentação final, relatório e script de inferência em produção

## 📈 Resultados

_A preencher conforme o projeto avança._

| Modelo | Split | AUC-ROC | Acurácia | F1 |
|---|---|---|---|---|
| RF + ECFP4 (baseline) | scaffold | — | — | — |
| SVM + ECFP4 (baseline) | scaffold | — | — | — |
| GCN | scaffold | — | — | — |
| GAT | scaffold | — | — | — |

## 🔗 Projetos Relacionados

- [Projeto 1 — QSAR Clássico (BTK)](https://github.com/GiovannaBorgesBasso/qsar-classico)

## 📚 Referências

- Martins et al. (2012), *A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling* — fonte original do dataset BBBP.
- Wu et al. (2018), *MoleculeNet: A Benchmark for Molecular Machine Learning*.
- Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks* (GCN).
- Veličković et al. (2018), *Graph Attention Networks* (GAT).
- Documentação oficial: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
