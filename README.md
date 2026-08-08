# Suavização de Rótulos em Redes Neurais: Estudo no Dataset UCI Bank Marketing

Repositório de pesquisa experimental em PyTorch desenvolvido para investigar como a Suavização de Rótulos ($\epsilon \in [0.0, 0.45]$) altera:
1. Valores Brutos de Logits ($z$ antes da Softmax): Prevenção de magnitudes extremas de ativação e variância delimitada.
2. Separação de Margens de Logits: Limitação de $(z_{\text{correto}} - z_{\text{outro}})$ para evitar explosão da norma dos pesos.
3. Generalização e Calibração: Avaliação de Acurácia, Precisão, Recall, F1-Score, AUC-ROC e Erro Esperado de Calibração (ECE) no conjunto de dados UCI Bank Marketing (45211 registros).
4. Dinâmica de Treinamento e Desempenho de Classificação: Curvas de Loss de Treino e Validação ao longo das épocas, comparativo ROC-AUC e Matrizes de Confusão no Teste.

## Mecânica Matemática ($\epsilon \in \{0.0, 0.15, 0.30, 0.45\}$)

Para classificação binária ($y \in \{0, 1\}$) com fator de suavização $\epsilon$:

$$y_{\text{suave}} = (1 - \epsilon) \cdot y + 0.5 \cdot \epsilon$$

| $\epsilon$ | Alvo $y=1$ ($y_{\text{suave}}$) | Alvo $y=0$ ($y_{\text{suave}}$) | Confiança Máxima Teórica |
| :--- | :--- | :--- | :--- |
| $\epsilon = 0.00$ (Baseline) | $1.000$ ($100.0\%$) | $0.000$ ($0.0\%$) | $\approx 100.0\%$ (Superconfiante) |
| $\epsilon = 0.15$ | $0.925$ ($92.5\%$) | $0.075$ ($7.5\%$) | $\approx 92.5\%$ |
| $\epsilon = 0.30$ | $0.850$ ($85.0\%$) | $0.150$ ($15.0\%$) | $\approx 85.0\%$ |
| $\epsilon = 0.45$ | $0.775$ ($77.5\%$) | $0.225$ ($22.5\%$) | $\approx 77.5\%$ |

## Estrutura do Repositório

```text
Smoothing_NN/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── data.py              # Carregamento e pré-processamento do UCI Bank Marketing (45.2k linhas)
│   ├── model.py             # MLP Tabular (128, 64, 32 com LayerNorm, ReLU e Dropout)
│   ├── losses.py            # Função de Perda Label Smoothing Cross-Entropy
│   ├── calibration.py       # Cálculo de métricas (Acc, Precisão, Recall, F1, ECE, Brier)
│   └── visualize.py         # Visualizações de calibração e desempenho de treino
│
└── main.py                  # Executor do benchmark em Full-Batch (sem mini-batches)
```

## Como Executar

### Executar Experimento de Benchmark em Full-Batch (500 Épocas + Early Stopping + Balanceamento de Classes)

```bash
# Padrão: Random Oversampling (balanceia treino para 50/50)
python main.py

# Alternativa: Ponderação de Perda Sensível ao Custo
python main.py --balance_method weights

# Ou sem balanceamento
python main.py --balance_method none
```

## Saídas Visuais Geradas

1. `results/label_smoothing_study.png`:
   * Linha 1: Histogramas dos Scores Brutos (Logits $z$) Antes da Softmax com Std e [min, max].
   * Linha 2: Histogramas das Margens de Logits ($z_{\text{correto}} - z_{\text{outro}}$).

2. `results/training_and_metrics.png`:
   * Painel 1: Curvas de Loss de Treino e Validação ao longo das épocas.
   * Painel 2: Curvas ROC com AUC no conjunto de Teste.
   * Painel 3: Matrizes de Confusão para cada $\epsilon \in \{0.0, 0.15, 0.30, 0.45\}$.

3. `results/threshold_table.png`:
   * Tabela Visual Executiva: Mapeamento de Cutoff (T), Recall (Captação), Clientes Contatados (% base), Convertidos e Taxa de Conversão (%) para o 4º Algoritmo ($\epsilon = 0.45$).

4. `results/threshold_histograms.png`:
   * Painéis de Histogramas de Probabilidade: Visualização da distribuição das probabilidades com destaque colorido para barras com $P \geq \text{Cutoff}$, detalhando Recall, Clientes Contatados e Taxa de Conversão para cada limiar avaliado no 4º Algoritmo ($\epsilon = 0.45$).
