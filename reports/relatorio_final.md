# Relatório Final — Projeto 2: GNN para Predição de BBBP

**Autora:** Giovanna Borges Basso  
**Laboratório:** LabMol (UFG)  
**Dataset:** MoleculeNet BBBP (~2.039 moléculas)  
**Repositório:** [projeto2-gnn-bbbp](https://github.com/GiovannaBorgesBasso/projeto2-gnn-bbbp)

---

## 1. Objetivo

Construir e avaliar um pipeline com Graph Neural Networks (GNN) para prever a permeabilidade da barreira hematoencefálica (BBB) de compostos químicos, comparando a representação aprendida por grafos com o baseline clássico de fingerprints + modelos estatísticos do Projeto 1.

A pergunta central: **a representação aprendida pela GNN supera fingerprints fixos (ECFP4) nessa tarefa específica?**

---

## 2. Dataset

- **Fonte:** MoleculeNet BBBP (Martins et al., 2012)
- **Tamanho:** 2.039 moléculas válidas
- **Tarefa:** classificação binária (BBB+ = atravessa a barreira / BBB- = não atravessa)
- **Desbalanceamento:** 76.5% BBB+ / 23.5% BBB-

### Características moleculares (EDA)

| Descritor | Mínimo | Mediana | Máximo |
|---|---|---|---|
| Átomos pesados | 2 | 23 | 132 |
| Peso molecular (Da) | 28 | 324 | 1.880 |

A maioria das moléculas está dentro do espaço farmacêutico clássico (MW < 500 Da, regra de Lipinski), esperado para um dataset focado em compostos CNS.

---

## 3. Decisões Metodológicas

### 3.1 Representação molecular

**Baseline clássico:** ECFP4 (Morgan fingerprint, raio 2, 2048 bits). Representação fixa extraída manualmente — cada molécula vira um vetor de presença/ausência de subestruturas circulares.

**GNN:** grafo molecular onde átomos são nós e ligações são arestas. Features de nó (6 por átomo): número atômico, grau, número de H ligados, carga formal, aromaticidade, hibridização. Features de aresta (3 por ligação): tipo de ligação, pertence a anel, é conjugada.

### 3.2 Protocolo de avaliação

**Split:** scaffold split estratificado (80/10/10 aproximado). O scaffold split padrão do BBBP resulta em conjuntos de validação e teste com 0 moléculas BBB-, tornando AUC-ROC indefinida. A versão estratificada garante ambas as classes em todos os conjuntos, mantendo a filosofia de não vazar scaffolds entre treino e teste.

**Métrica principal:** AUC-ROC — insensível ao desbalanceamento de classes (76/24), ao contrário da acurácia.

**Desbalanceamento:** tratado via `class_weight='balanced'` nos modelos clássicos e `BCEWithLogitsLoss(pos_weight=3.29)` nas GNNs.

---

## 4. Modelos

### 4.1 Baseline clássico

| Modelo | Implementação |
|---|---|
| Random Forest | `sklearn.ensemble.RandomForestClassifier`, 100 árvores, `class_weight='balanced'` |
| SVM | `sklearn.svm.SVC`, kernel RBF, `class_weight='balanced'` |

### 4.2 GCN (Graph Convolutional Network)

- 3 camadas `GCNConv` com hidden_channels=64
- BatchNorm + ReLU + Dropout (0.3) após cada camada
- Global Mean Pooling para agregar átomos → molécula
- Classificador linear final
- 9.217 parâmetros treináveis
- Treinada por 68 épocas (early stopping)

### 4.3 GAT (Graph Attention Network)

- 3 camadas `GATConv` com hidden_channels=128, 4 cabeças de atenção
- BatchNorm + ReLU + Dropout (0.3) após cada camada
- Global Mean Pooling
- Classificador linear final
- 35.585 parâmetros treináveis
- Treinada por 54 épocas (early stopping)

**Diferença principal GCN vs GAT:** a GCN agrega vizinhos com pesos fixos baseados na estrutura do grafo. A GAT aprende coeficientes de atenção durante o treino — cada átomo pondera seus vizinhos de forma diferente dependendo do contexto.

---

## 5. Resultados

| Modelo | AUC-ROC (teste) |
|---|---|
| Random Forest + ECFP4 | 0.8420 |
| **SVM + ECFP4** | **0.8564** |
| GCN (grafos) | 0.8366 |
| GAT (grafos) | 0.8419 |

![Comparação de modelos](../results/figures/comparacao_modelos.png)

![Curvas de treino](../results/figures/curvas_treino.png)

---

## 6. Análise e Discussão

### Os modelos clássicos se saíram melhor. Por quê?

O resultado é consistente com a literatura sobre o BBBP. Dois fatores explicam a vantagem dos fingerprints nessa tarefa:

**Tamanho do dataset.** Com ~2.000 moléculas, as GNNs não têm dados suficientes para aprender representações que superem features cuidadosamente projetadas como ECFP4. GNNs geralmente superam fingerprints a partir de ~10.000 moléculas.

**Natureza da tarefa.** A permeabilidade na BBB é fortemente correlacionada com propriedades globais simples (lipofilia, número de doadores/aceptores de H-bond, peso molecular) — exatamente o tipo de informação que ECFP4 captura bem. A topologia molecular detalhada que as GNNs exploram agrega menos valor nesse caso.

### A GNN não falhou — empatou

A GAT (0.842) empatou com o Random Forest (0.842) e ficou 1.4 pontos percentuais abaixo do SVM. Não é uma derrota expressiva. O resultado mostra que GNNs são competitivas mesmo em regimes de poucos dados, mas não justificam a complexidade adicional nesse cenário específico.

### O que mudaria com mais dados?

A literatura mostra que GNNs pré-treinadas em milhões de moléculas (como os modelos do paper Hu et al., 2020 — "Strategies for Pre-Training Graph Neural Networks") superam consistentemente fingerprints no BBBP. O pré-treino resolve o problema de dados escassos ao transferir conhecimento aprendido em datasets maiores.

---

## 7. Limitações

- **Dataset pequeno:** 2.039 moléculas é insuficiente para GNNs mostrarem seu potencial máximo.
- **Sem pré-treino:** GNNs foram treinadas do zero. Pré-treino em datasets maiores provavelmente inverteria o resultado.
- **Hiperparâmetros não otimizados:** um grid search ou Bayesian optimization poderia melhorar ambas as GNNs.
- **Conjunto de validação pequeno:** 174 moléculas (8.5%) gera val_auc instável durante o treino, dificultando o early stopping.

---

## 8. Próximos Passos

- **Etapa 9 (planejada):** replicar o pipeline no B3DB (~7.807 moléculas) para verificar se as GNNs superam fingerprints com mais dados.
- **Pré-treino:** usar modelos GNN pré-treinados (ex: via DeepChem ou HuggingFace) com fine-tuning no BBBP.
- **Otimização de hiperparâmetros:** grid search sobre hidden_channels, num_layers, dropout, learning rate.
- **Features adicionais:** adicionar quiralidade e estereoquímica de aresta ao featurizador.

---

## 9. Referências

- Martins et al. (2012). *A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling.* Journal of Chemical Information and Modeling.
- Wu et al. (2018). *MoleculeNet: A Benchmark for Molecular Machine Learning.* Chemical Science.
- Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
- Veličković et al. (2018). *Graph Attention Networks.* ICLR.
- Hu et al. (2020). *Strategies for Pre-Training Graph Neural Networks.* ICLR.
