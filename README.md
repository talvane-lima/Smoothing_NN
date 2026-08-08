# Label Smoothing in Neural Networks: UCI Bank Marketing Study

An experimental research repository in PyTorch designed to investigate how Label Smoothing ($\epsilon \in [0.0, 0.45]$) alters:
1. Raw Logit Scores ($z$ before Softmax): Preventing extreme activation magnitudes and bounded variance.
2. Logit Margin Separation: Bounding $(z_{\text{correct}} - z_{\text{other}})$ to prevent exploding weight norms.
3. Generalization and Calibration: Evaluating Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Expected Calibration Error (ECE) on the UCI Bank Marketing dataset (45211 records).
4. Training Dynamics and Classification Performance: Training and Validation Loss curves over epochs, ROC-AUC comparison, and Test Confusion Matrices.

## Mathematical Mechanics ($\epsilon \in \{0.0, 0.15, 0.30, 0.45\}$)

For binary classification ($y \in \{0, 1\}$) with smoothing factor $\epsilon$:

$$y_{\text{smooth}} = (1 - \epsilon) \cdot y + 0.5 \cdot \epsilon$$

| $\epsilon$ | Target $y=1$ ($y_{\text{smooth}}$) | Target $y=0$ ($y_{\text{smooth}}$) | Theoretical Max Confidence |
| :--- | :--- | :--- | :--- |
| $\epsilon = 0.00$ (Baseline) | $1.000$ ($100.0\%$) | $0.000$ ($0.0\%$) | $\approx 100.0\%$ (Overconfident) |
| $\epsilon = 0.15$ | $0.925$ ($92.5\%$) | $0.075$ ($7.5\%$) | $\approx 92.5\%$ |
| $\epsilon = 0.30$ | $0.850$ ($85.0\%$) | $0.150$ ($15.0\%$) | $\approx 85.0\%$ |
| $\epsilon = 0.45$ | $0.775$ ($77.5\%$) | $0.225$ ($22.5\%$) | $\approx 77.5\%$ |

## Repository Structure

```text
Smoothing_NN/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── data.py              # Full-Tensor loader and preprocessor for UCI Bank Marketing (45.2k rows)
│   ├── model.py             # Tabular MLP (128, 64, 32 with LayerNorm, ReLU and Dropout)
│   ├── losses.py            # Label Smoothing Cross-Entropy Loss
│   ├── calibration.py       # Metrics computation (Acc, Precision, Recall, F1, ECE, Brier)
│   └── visualize.py         # Visualizers for calibration and training performance
│
└── main.py                  # Full-Batch benchmark runner (no mini-batches)
```

## How to Run

### Run Full-Batch Benchmark Experiment (500 Epochs + Early Stopping + Class Balancing)

```bash
# Default: Random Oversampling (balances training set to 50/50)
python main.py

# Alternatively: Cost-Sensitive Loss Weighting
python main.py --balance_method weights

# Or without balancing
python main.py --balance_method none
```

## Visual Outputs Generated

1. `results/label_smoothing_study.png`:
   * Linha 1: Histogramas dos Scores Brutos (Logits $z$) Antes da Softmax com Std e [min, max].
   * Linha 2: Histogramas das Margens de Logits ($z_{\text{correto}} - z_{\text{outro}}$).

2. `results/training_and_metrics.png`:
   * Painel 1: Curvas de Loss de Treino e Validacao ao longo das epocas.
   * Painel 2: Curvas ROC com AUC no conjunto de Teste.
   * Painel 3: Matrizes de Confusao para cada $\epsilon \in \{0.0, 0.15, 0.30, 0.45\}$.

3. `results/threshold_table.png`:
   * Tabela Visual Executiva: Mapeamento de Cutoff (T), Recall (Captacao), Clientes Contatados (% base), Convertidos e Taxa de Conversao (%) para o 4o Algoritmo ($\epsilon = 0.45$).
