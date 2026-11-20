# Logistic Regression for Attack Detection in Computer Networks

Logistic regression **implemented from scratch with NumPy** (no scikit-learn) that classifies network flows as benign or malicious, using the CICIDS 2017 dataset.

## Overview

Each row in the dataset is a network flow described by statistics extracted by CICFlowMeter (packet counts, byte counts, inter-arrival times, flag counts, etc.). The model is trained with batch gradient descent to predict whether a flow is `BENIGN` (0) or an attack (1).

## Dataset

The project uses the **CICIDS 2017** CSV files (`MachineLearningCVE` / `*.pcap_ISCX.csv`). Download them from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) and put them in the `data/` folder, which is excluded from version control via `.gitignore`.

| File | Rows | Labels | Suitability |
|---|---|---|---|
| `Friday-WorkingHours-Afternoon-DDos` | 225 745 | DDoS 128 027 / BENIGN 97 718 | **Recommended**: balanced classes |
| `Friday-WorkingHours-Afternoon-PortScan` | 286 467 | PortScan 158 930 / BENIGN 127 537 | Good, but ~25% duplicate rows |
| `Friday-WorkingHours-Morning` | 191 033 | Bot 1 966 / BENIGN 189 067 | Highly imbalanced |
| `Thursday-WorkingHours-Morning-WebAttacks` | 170 366 | Brute Force 1 507, XSS 652, SQL Injection 21 / BENIGN 168 186 | Imbalanced (~1.3% attacks), hard for flow features |
| `Thursday-WorkingHours-Afternoon-Infilteration` | 288 602 | Infiltration 36 / BENIGN 288 566 | **Not usable**: only 36 attacks |
| `Monday-WorkingHours` | 529 918 | BENIGN only | No attacks |

> [!WARNING]
> With highly imbalanced files (WebAttacks, Infiltration, Bot), a model that always predicts `BENIGN` reaches 98–99.99% accuracy. Judge those by precision and recall of the attack class, not by accuracy.

> [!NOTE]
> The `Thursday-...-WebAttacks` file contains a non-UTF-8 character in its labels. If `pd.read_csv` fails on it, add `encoding="latin1"` in `clean_data`.

## How It Works

The code is split into three files.

### `data_manipulating.py`

- **`clean_data(path)`**
  - Loads the CSV and strips whitespace from column names.
  - Replaces `inf`/`-inf` with `NaN` and drops all rows containing `NaN`.
  - Drops identifier and shortcut columns if present: `Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Destination Port`, `Timestamp` and `Subflow Fwd Bytes`. Ports are removed so the model cannot rely on them as a shortcut.
  - Drops constant columns (a single unique value).
  - Encodes the label: `BENIGN` → `0`, anything else → `1`. All attack types are merged into one class.
  - Returns the feature table `x` and the labels `y`.
- **`split_data(x, y, test_size, seed)`**: shuffles indices with a seeded `numpy` generator and splits into train/test. The split is random, not stratified.
- **`normalize(x_training, x_test)`**: standardizes features (z-score) using the mean and standard deviation of the **training set only**, so no test information leaks. Zero standard deviations are replaced with 1.

### `log_reg.py`

- **`sigmoid(z)`**: the logistic function, with `z` clipped to `[-500, 500]` to avoid overflow.
- **`compute_loss(y, y_hat)`**: binary cross-entropy, with predictions clipped to avoid `log(0)`.
- **`predict(x, weights, b)`**: returns the attack probability for each row.
- **`train(x, y, lr, epochs)`**: full-batch gradient descent starting from zero weights. It prints the loss every 100 epochs and returns `(weights, b)`. There are no class weights and no regularization.
- **`compare(x, y, weights, b, threshold=0.5)`**: prints the number of real and detected attack and benign flows, accuracy, precision and recall.

### `test.py`

The entry script. It runs the full pipeline: clean → split (70/30, `seed=42`) → normalize → train (`lr=0.5`, `epochs=1000`) → evaluate on the test set.

`CSV_PATH` at the top of `test.py` currently points to the PortScan file. Change it to the file you want to use. Use forward slashes or a raw string to avoid backslash escapes on Windows, for example `r"data\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"`.

## Project Structure

```
logistics-regression-in-computer-networks/
├── data/                   # CICIDS 2017 CSV files (gitignored)
├── data_manipulating.py    # Loading, cleaning, splitting, normalization
├── log_reg.py              # Sigmoid, loss, training, prediction, evaluation
├── test.py                 # Entry point running the whole pipeline
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- `numpy`
- `pandas`

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS
pip install numpy pandas
```

## Usage

```bash
python test.py
```

### Sample Output

```
Epoch: 0, Loss: ...
Epoch: 100, Loss: ...
...
Bad packets: N, detected: N
Normal packets: N, detected: N
Accuracy: 0.xxxx
Precision: 0.xxxx, Recall: 0.xxxx
```

## Known Limitations

- Duplicate rows are not removed, so identical flows can land in both train and test and inflate the results (about 3.6% in WebAttacks, 25% in PortScan).
- The train/test split is random, not stratified or time-based. Flows from the same attack are very similar, so the results are optimistic.
- All attack types are merged into one class, so the model cannot tell attacks apart.
- No class weighting, so heavily imbalanced files give poor attack recall.

## References

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP.
- [CICIDS 2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
