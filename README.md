<div align="center">

# Linear Regression from Scratch

A pure NumPy implementation uncovering the internal mechanics of linear modeling, cost functions, and first-order gradient optimization—without relying on black-box frameworks like scikit-learn.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243.svg?logo=numpy&logoColor=white)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualizations-11557c.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Architecture](#-learning-pipeline) • [Mathematics](#-mathematical-foundation) • [Project Structure](#-project-structure) • [Quickstart](#-quickstart) • [Evaluation](#-evaluation-metrics)

</div>

---

## Overview

High-level libraries like `scikit-learn` abstract away model optimization into single-line calls like `.fit()`. This project strips away the abstraction layer to implement every mathematical component directly in **NumPy**:

- Analytical prediction via linear combinations
- Cost surface mapping via Mean Squared Error (MSE)
- Closed-form partial derivative calculations
- Batch gradient descent optimization routines
- Per-epoch loss tracking & diagnostic convergence plotting
- Out-of-sample inference and comprehensive statistical validation

---

## Learning Pipeline

The training loop proceeds iteratively across defined epochs:

---

## Mathematical Foundation

### 1. Hypothesis Function
The univariate model maps an input feature vector $X$ to target predictions $\hat{y}$:

$$\hat{y} = w \cdot x + b$$

| Parameter | Type | Description |
| :--- | :--- | :--- |
| $w$ | Scalar Weight | Slope / feature coefficient |
| $b$ | Scalar Bias | Vertical intercept term |
| $x$ | Feature Input | Regressor value |
| $\hat{y}$ | Target Estimate | Output variable prediction |

### 2. Loss Function (Mean Squared Error)
The optimization objective seeks to minimize the convex objective function:

$$J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 3. Gradient Derivations
Using the chain rule, partial derivatives with respect to parameters $w$ and $b$ are computed analytically:

$$\frac{\partial J}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} x_i (\hat{y}_i - y_i)$$

$$\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

### 4. Parameter Update Rule
Weights and biases update in the negative direction of the loss gradient scaled by learning rate $\alpha$:

$$w \leftarrow w - \alpha \cdot \frac{\partial J}{\partial w}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}$$

---

## Project Structure

```text
linear-regression-from-scratch/
├── model.py            # LinearRegressionScratch class (forward pass, loss, backward pass)
├── train.py            # Pipeline driver (data creation, training loop, evaluation, plotting)
├── requirements.txt    # Runtime dependencies (numpy, matplotlib)
├── .gitignore          # Environment & bytecode filter rules
└── README.md           # Technical documentation
```

