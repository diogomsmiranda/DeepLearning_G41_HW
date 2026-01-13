# Deep Learning Homework Repository

This repository contains implementations of fundamental deep learning models using **NumPy** and **PyTorch**, developed for the Deep Learning course (DEI).

**Authors:** Rafael Sargento, Diogo Miranda

---

## Structure

### Homework 1: Linear Classifiers & MLPs
* **`hw1-q1.py`**: Implementation of Linear Models (Perceptron, Logistic Regression) and a Multi-Layer Perceptron (MLP) from scratch using **NumPy**.
* **`hw1-q2.py`**: Implementation of Logistic Regression and Feedforward Networks using **PyTorch**.
* **`hw1_RS_DM.pdf`**: Report containing analysis, plots, and derivation answers.

### Homework 2: CNNs & RNNs (Seq2Seq)
* **`hw2-q2.py`**: Implementation of Convolutional Neural Networks (CNN) with configurable blocks (Dropout, BatchNorm, MaxPool).
* **`hw2-q3.py`**: Training and testing script for Sequence-to-Sequence models (Grapheme-to-Phoneme).
* **`models.py`**: PyTorch architecture definitions for HW2 Q3 (Encoder, Decoder, Seq2Seq, Bahdanau Attention).
* **`data.py`**: Data preprocessing, vocabulary handling, and tokenization for Seq2Seq.
* **`hw2_RS_DM.pdf`**: Report containing analysis, plots, and theoretical answers.

### Shared Utilities
* **`utils.py`**: Helper functions for random seeding and dataset loading. (Courses's Given Functions)

---

## Requirements

* Python 3.x
* PyTorch
* NumPy
* Matplotlib

---

## Usage

### Homework 1

**Run NumPy Models (Q1):**
```bash
# Options: perceptron, logistic_regression, mlp
python hw1-q1.py mlp -epochs 20 -learning_rate 0.001
```

**Run PyTorch Models (Q2):**
```bash
# Options: logistic_regression, mlp
python hw1-q2.py mlp -epochs 50 -dropout 0.3 -optimizer adam
```

### Homework 2

**Run CNN (Q2):**
```bash
python hw2-q2.py -epochs 40 -learning_rate 0.01 -dropout 0.1
```

**Run Seq2Seq (Q3):**
```bash
# Train simple Seq2Seq
python hw2-q3.py train --n_epochs 20

# Train Seq2Seq with Attention
python hw2-q3.py train --use_attn --n_epochs 20

# Test a model (Greedy decoding)
python hw2-q3.py test --checkpoint_name model.pt

# Test with Nucleus Sampling (Top-p)
python hw2-q3.py test --checkpoint_name model.pt --topp 0.9
```
