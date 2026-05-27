# RAMHGN

Code and datasets for **RAMHGN: A Relation-aware Multi-Scale Heterogeneous Graph Network for Drug-Disease Association Prediction**.

RAMHGN is designed for drug-disease association prediction. It constructs heterogeneous biological networks from drug, disease and related biomedical relations, then learns relation-aware and multi-scale representations for association prediction.

## Requirements

Recommended environment:

```bash
conda create -n ramhgn python=3.10.14
conda activate ramhgn
```

Install dependencies:

```bash
pip install torch==2.5.1
pip install "dgl>=1.1.2"
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

If you use GPU, please install the PyTorch and DGL versions that match your CUDA version.

## Datasets

The datasets used in this project are stored in `dataset/`:

```text
dataset/
+-- Kdataset/
+-- Fdataset/
+-- Cdataset/
```

### Kdataset

Main files:

- `dataset/Kdataset/Kdataset_baseline.csv`
- `dataset/Kdataset/drug_drug_baseline.csv`
- `dataset/Kdataset/disease_disease_baseline.csv`
- `dataset/Kdataset/associations/`
- `dataset/Kdataset/interactions/`
- `dataset/Kdataset/omics/`

### Fdataset

Main files:

- `dataset/Fdataset/drug_disease.csv`
- `dataset/Fdataset/drug_drug.csv`
- `dataset/Fdataset/disease_disease.csv`
- `dataset/Fdataset/drug.csv`
- `dataset/Fdataset/disease.csv`

### Cdataset

Main file:

- `dataset/Cdataset/Cdataset.mat`

## Project Structure

```text
.
+-- args.py
+-- main.py
+-- model.py
+-- load_data.py
+-- utils.py
+-- default_parameters.md
+-- dataset/
```

## Run

Default training on `Kdataset`:

```bash
python main.py
```

Run on different datasets:

```bash
python main.py -da Kdataset -sp resultK
python main.py -da Fdataset -sp resultF
python main.py -da Cdataset -sp resultC
```

Set GPU device:

```bash
python main.py -id 0 -da Kdataset -sp resultK
```

Common training options:

```bash
python main.py \
  -da Kdataset \
  -fo 10 \
  -ep 4000 \
  -lr 0.003 \
  -wd 0.0001 \
  -pa 300 \
  -hf 128 \
  -he 5 \
  -dp 0.5
```

Important arguments:

- `-da, --dataset`: dataset name, including `Kdataset`, `Fdataset`, `Cdataset`.
- `-sp, --saved_path`: path for saving results.
- `-se, --seed`: random seed.
- `-fo, --nfold`: number of folds for cross-validation.
- `-ep, --epoch`: maximum training epochs.
- `-lr, --learning_rate`: learning rate.
- `-hf, --hidden_feats`: hidden feature dimension.
- `-he, --num_heads`: number of attention heads.
- `-dp, --dropout`: dropout rate.

More adjustable parameters are listed in `default_parameters.md`.

## Output

The result directory is named by `--saved_path` and `--seed`. For example, `-sp resultK -se 42` saves results to:

```text
resultK_42/
```

Main outputs:

- `early_stop_*.pth`: saved model checkpoints.
- `result.csv`: predicted drug-disease association scores.
- `result_auc.png`: ROC curve.
- `result_aupr.png`: PR curve.

The program reports AUROC, AUPR, Accuracy, F1-score, Precision, Recall and Specificity during evaluation.
