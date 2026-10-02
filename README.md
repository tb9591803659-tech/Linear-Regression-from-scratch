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

## Module Responsibilities

| File | Primary Functions / Classes | Core Responsibility |
| :--- | :--- | :--- |
| `model.py` | `LinearRegressionScratch` | Encapsulates parameters ($w, b$), learning rate, batch updates, and cost history. |
| `train.py` | `main()` | Orchestrates synthetic dataset generation, training execution, evaluation, and visualizations. |

## Evaluation Metrics

Model fit and predictive fidelity are measured against four standard regression statistics:

| Metric | Formulation | Interpretation |
| :--- | :--- | :--- |
| **Mean Squared Error (MSE)** | $\frac{1}{n} \sum (y - \hat{y})^2$ | Penalizes larger residuals quadratically; measures variance. |
| **Mean Absolute Error (MAE)** | $\frac{1}{n} \sum \|y - \hat{y}\|$ | Direct linear average of absolute prediction discrepancies. |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\text{MSE}}$ | Error metric expressed in the original units of target $y$. |
| **Coefficient of Determination ($R^2$)** | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ | Proportion of total variance explained by model (1.0 = perfect fit). |


## Quickstart

### Prerequisites

- Python 3.8 or higher

### 1. Clone and enter the project

```bash
git clone https://github.com/<your-username>/linear-regression-from-scratch.git
cd linear-regression-from-scratch
```

### 2. Set up a virtual environment and install dependencies

```bash
# Create an isolated virtual environment
python -m venv venv

# Activate it
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the training pipeline

```bash
python train.py
```

## Sample Run & Expected Output

### Training Data Setup

```python
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])  # Follows y = 2x + 0
```

### Terminal Output

```text
[Training Completed: 1000 Epochs]
Learned Parameters:
  • Weight (w) : 1.9998  (Target: 2.0)
  • Bias (b)   : 0.0006  (Target: 0.0)

Evaluation on Training Data:
  • MSE  : 0.0000003
  • MAE  : 0.0004
  • RMSE : 0.0005
  • R²   : 0.9999999

Out-of-Sample Predictions (X_new = [6, 7, 8]):
  • Input: 6  -->  Predicted: 11.999
  • Input: 7  -->  Predicted: 13.999
  • Input: 8  -->  Predicted: 15.999
```

### Output Visualizations

Running `train.py` renders two diagnostic figures:

- **Best-Fit Regression Line:** Overlays the learned line $\hat{y} = wx + b$ on the original training data points.
- **Cost Convergence Curve:** Plots $J(w, b)$ over all iterations to confirm the cost descends asymptotically toward zero.

## Scratch vs. Library Implementation

| Aspect | `sklearn.linear_model.LinearRegression` | This Scratch Implementation |
|---|---|---|
| **Solving Method** | Closed-form Ordinary Least Squares (`scipy.linalg.lstsq`) | Numerical optimization (Batch Gradient Descent) |
| **Hyperparameters** | None (direct matrix factorization) | Learning rate ($\alpha$), total epochs |
| **Observability** | Final coefficients only (`.coef_`, `.intercept_`) | Full step-by-step state, per-epoch loss, and gradient traces |
| **Primary Purpose** | Production efficiency and scale | Pedagogical clarity and algorithmic understanding |

## Roadmap

- [ ] Support for Multiple Linear Regression ($X \in \mathbb{R}^{n \times d}$)
- [ ] Z-score feature standardization and min-max normalization utilities
- [ ] L1 (Lasso) and L2 (Ridge) weight regularization penalties
- [ ] Mini-batch and Stochastic Gradient Descent (SGD) optimizers

## License

This project is licensed under the MIT License. See the [MIT License](https://spdx.org/licenses/MIT?utm_source=chatgpt.com) file for details.