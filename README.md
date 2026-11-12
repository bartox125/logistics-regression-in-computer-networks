# Logistic Regression for DDoS Attack Detection in Computer Networks

A machine learning project that applies **Logistic Regression** to classify network traffic as either benign or a DDoS attack, using the CICIDS 2017 dataset.

## Overview

This project demonstrates how logistic regression can be used to detect DDoS attacks in network traffic. The model is trained on flow-level features extracted from network packet captures (`.pcap` files), processed using the CICFlowMeter tool.

## Dataset

The project uses the **CICIDS 2017** dataset — specifically the Friday afternoon DDoS traffic capture:

```
data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

> [!NOTE]
> The `data/` directory is excluded from version control via `.gitignore`. Download the dataset from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) and place the CSV file in the `data/` folder.

The dataset contains labelled network flow records with the target column `Label`:
- `BENIGN` → encoded as `0`
- DDoS attack traffic → encoded as `1`

## How It Works

1. **Data loading & cleaning** — Loads the CSV, strips whitespace from column names, removes infinite values and NaN rows.
2. **Feature selection** — Drops non-informative columns (`Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Destination Port`, `Timestamp`, `Subflow Fwd Bytes`) and constant-value columns.
3. **Label encoding** — Binary encoding: `BENIGN=0`, attack=`1`.
4. **Train/test split** — 70% training / 30% test, stratified by label, `random_state=42`.
5. **Feature scaling** — `StandardScaler` applied to normalize features.
6. **Model training** — `LogisticRegression` with `class_weight='balanced'` to handle class imbalance.
7. **Evaluation** — Outputs a **Confusion Matrix** and a full **Classification Report** (precision, recall, F1-score).
8. **Feature importance** — Lists the top 5 most influential features based on model coefficients.

## Project Structure

```
logistics-regression-in-computer-networks/
├── data/                   # Dataset directory (gitignored)
│   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── main.py                 # Main script
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `scikit-learn`

Install dependencies:

```bash
pip install numpy pandas scikit-learn
```

Or, if using a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS
pip install numpy pandas scikit-learn
```

## Usage

```bash
python main.py
```

### Sample Output

```
Training data shape: (N, features)
Number of attacks in training set: ...
Number of normal traffic records: ...

=== Confusion Matrix ===
[[TN  FP]
 [FN  TP]]

=== Classification Report ===
              precision    recall  f1-score   support
  Normal (0)       ...
  Attack (1)       ...

=== TOP 5 MOST IMPORTANT FEATURES ===
              Feature    Weight
...
```

## References

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP.
- [CICIDS 2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
