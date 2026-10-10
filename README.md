# Predicting Metabolic Cost During Human-in-the-Loop Optimization

## Overview

This project reproduces and extends selected experiments from the CS229 paper **“Predicting Metabolic Cost During Human-in-the-Loop Optimization”** by Eley Ng and Erez Krimsky (2018).

The goal is to predict the metabolic cost of human movement using biomechanical and electromyography (EMG) features. We implement and evaluate multiple machine learning models and investigate the effect of different feature configurations.

The project includes:

- LASSO regression for regularized linear prediction
- Neural network regression
- Random Forest regression as an extension
- Forward stepwise feature selection
- Principal Component Analysis (PCA)
- Five-fold cross-validation for model selection and evaluation

This project is an **approximate reproduction**, rather than an exact replication, of the original paper. Some implementation details differ from the authors’ methodology.

## Paper Reference

Eley Ng and Erez Krimsky, *Predicting Metabolic Cost During Human-in-the-Loop Optimization*, 2018.

The original study investigates whether metabolic cost can be predicted from movement-related features and compares linear regression and neural network models under different feature configurations.

## Project Structure

```text
Metabolic-Cost-Prediction-ML-mini-project/
├── data/
│   └── processed_data.mat
├── results/
│   └── model_results.txt
├── src/
│   ├── evaluation.py
│   ├── feature_selection.py
│   ├── linear_regression.py
│   ├── load_data.py
│   ├── neural_network.py
│   ├── preprocessing.py
│   ├── random_forest.py
│   ├── run_feature_selection.py
│   ├── run_linear.py
│   ├── run_nn.py
│   └── run_random_forest.py
├── requirements.txt
└── README.md
```

## Dataset and Features

The project uses the processed dataset stored in `data/processed_data.mat`.

The experiments consider two subjects and evaluate four feature configurations:

| Configuration | Description |
|---|---|
| `all` | All available features |
| `step_emg` | Step-related and EMG features |
| `step_only` | Step-related features only |
| `emg_only` | EMG features only |

The preprocessing pipeline standardizes the input features. Subject 1 is additionally filtered to retain samples with non-negative target values, as implemented in the project scripts.

## Methods

### 1. LASSO Regression

LASSO is used as a regularized linear regression baseline. It applies an L1 penalty to the regression coefficients, encouraging sparse solutions.

- Model: `sklearn.linear_model.Lasso`
- Alpha selection: search over 25 values between \(10^{-5}\) and \(10^{-1}\)
- Cross-validation: five-fold cross-validation
- Random seed: 42

The selected alpha and corresponding mean squared error (MSE) are recorded for each subject and feature configuration.

### 2. Neural Network Regression

A feedforward neural network is implemented using `sklearn.neural_network.MLPRegressor`.

The network uses:

- One hidden layer
- Hyperbolic tangent (`tanh`) activation
- L-BFGS optimization
- L2 regularization with `alpha=1.0`
- A search over 1–14 hidden-layer neurons
- Five-fold cross-validation

**Implementation note:** The paper uses Bayesian Regularization for neural network training. This implementation uses L2 regularization through scikit-learn instead, so the neural network is an approximation of the paper's method rather than an exact reproduction.

### 3. Random Forest Regression

Random Forest is included as an extension beyond the paper's principal model comparison.

The implementation uses:

- 200 decision trees
- No fixed maximum tree depth
- Random seed 42
- Parallel processing where available
- Five-fold cross-validation

Random Forest is evaluated using the same four feature configurations for both subjects. Its performance is compared with the LASSO and neural network results.

### 4. Feature Selection

Forward stepwise feature selection is implemented using linear regression and five-fold cross-validation.

The current experiment tests a sequence of 10 candidate features for Subject 1. The selected feature indices, in selection order, are:

```text
26, 18, 24, 2, 7, 3, 15, 8, 6, 13
```

This is a limited feature-selection experiment. It does not reproduce the paper's full repeated feature-selection procedure or its complete evaluation of selected subsets using a tuned neural network.

### 5. Principal Component Analysis (PCA)

PCA is used to examine dimensionality reduction while retaining at least 98.8% of the variance.

For Subject 1, the current implementation reports:

- Original features: 29
- PCA components retained: 24
- Explained variance: 99.14%

These results describe the dimensionality reduction performed by the current implementation. The PCA result is not a direct performance comparison with the paper's PCA experiment, because the component count and evaluation procedure differ.

## Experimental Setup

The experiments evaluate two subjects using the four feature configurations described above.

Model performance is measured using **mean squared error (MSE)**. Lower MSE indicates lower average squared prediction error.

The implemented scripts use five-fold cross-validation. The results below are the values recorded in `results/model_results.txt`.

Because preprocessing and model-selection details differ from the original study, results should be interpreted as a reproduction attempt and model comparison, not as an exact replication.

## Results

### LASSO Regression

| Subject | Features | Selected alpha | MSE |
|---|---|---:|---:|
| 1 | All | 0.010000 | 0.012260 |
| 1 | Step + EMG | 0.006813 | 0.014437 |
| 1 | Step only | 0.006813 | 0.019794 |
| 1 | EMG only | 0.006813 | 0.017389 |
| 2 | All | 0.002154 | 0.008769 |
| 2 | Step + EMG | 0.001468 | 0.009436 |
| 2 | Step only | 0.001468 | 0.019450 |
| 2 | EMG only | 0.004642 | 0.015786 |

### Neural Network

| Subject | Features | Hidden neurons | MSE |
|---|---|---:|---:|
| 1 | All | 9 | 0.014007 |
| 1 | Step + EMG | 11 | 0.011953 |
| 1 | Step only | 4 | 0.016173 |
| 1 | EMG only | 5 | 0.015813 |
| 2 | All | 5 | 0.008064 |
| 2 | Step + EMG | 13 | 0.009076 |
| 2 | Step only | 13 | 0.015513 |
| 2 | EMG only | 13 | 0.013591 |

### Random Forest

| Subject | Features | MSE |
|---|---|---:|
| 1 | All | 0.011943 |
| 1 | Step + EMG | 0.012271 |
| 1 | Step only | 0.015741 |
| 1 | EMG only | 0.013418 |
| 2 | All | 0.011319 |
| 2 | Step + EMG | 0.011045 |
| 2 | Step only | 0.015569 |
| 2 | EMG only | 0.013938 |

### Best Configuration for Each Model

The following table shows the lowest recorded MSE for each model and subject.

| Subject | Model | Best feature configuration | MSE |
|---|---|---|---:|
| 1 | LASSO | All | 0.012260 |
| 1 | Neural Network | Step + EMG | 0.011953 |
| 1 | Random Forest | All | 0.011943 |
| 2 | LASSO | All | 0.008769 |
| 2 | Neural Network | All | 0.008064 |
| 2 | Random Forest | Step + EMG | 0.011045 |

### Comparison with the Paper

The original paper reports the following MSE values for its linear regression and neural network models.

| Subject | Features | Paper: Linear Regression | Paper: Neural Network |
|---|---|---:|---:|
| 1 | All | 0.0089 | 0.0089 |
| 1 | Step + EMG | 0.0089 | 0.0104 |
| 1 | Step only | 0.0200 | 0.0138 |
| 1 | EMG only | 0.0158 | 0.0118 |
| 2 | All | 0.0176 | 0.0301 |
| 2 | Step + EMG | 0.0168 | 0.0313 |
| 2 | Step only | 0.0217 | 0.0233 |
| 2 | EMG only | 0.0203 | 0.0199 |

*Source: the original paper's Table I. The paper labels its linear model “Lin. Reg.”; its methods discuss regularized LASSO regression.*

The paper's reported results and this project's results are **not directly equivalent**, because the implementations and evaluation procedures differ. In particular, the neural network training method differs, the feature-selection experiment is limited, and preprocessing is performed before cross-validation in the current implementation. The tables are therefore presented as reference benchmarks rather than proof of an exact reproduction.

## Discussion

### LASSO

LASSO provides a regularized linear baseline. In the current results, using all features gives the lowest MSE among the tested LASSO configurations for both subjects.

### Neural Network

The neural network performs best with the step-plus-EMG configuration for Subject 1 and with all features for Subject 2. This suggests that the most useful feature configuration can vary across subjects.

### Random Forest

Random Forest is competitive for Subject 1 and obtains the lowest MSE among the three models for the all-feature, step-only, and EMG-only configurations. For Subject 2, the neural network and LASSO achieve lower MSE than Random Forest with all features and with step-plus-EMG features.

Random Forest therefore provides a useful extension, but the results do not show that it consistently outperforms the other models for both subjects.

### Feature Selection and PCA

The feature-selection and PCA experiments explore ways to reduce the input dimensionality. The current implementation provides preliminary results, but further evaluation is needed to determine whether these reductions improve prediction performance.

## Limitations

- The project is an approximate reproduction, not an exact replication of the paper.
- Neural network training uses L2 regularization rather than Bayesian Regularization.
- Forward stepwise selection tests only 10 features in the current experiment.
- PCA results are reported as explained variance and component count, without a corresponding reduced-feature model comparison in the recorded results.
- Standardization is performed before cross-validation, which can introduce data leakage because validation-fold information contributes to the scaling parameters.
- The reported MSE values depend on the implementation and evaluation setup and should not be assumed to match the paper's values exactly.

## Future Work

- Implement a training procedure closer to the paper's Bayesian Regularization approach.
- Fit preprocessing transformations separately within each cross-validation training fold.
- Expand feature-selection experiments and evaluate selected subsets with the neural network.
- Evaluate prediction performance after PCA-based dimensionality reduction.
- Repeat experiments across random seeds or repeated cross-validation splits to assess result stability.
- Add further evaluation metrics and visualizations to support model comparison.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/sharvanee08/Metabolic-Cost-Prediction-ML-mini-project.git
cd Metabolic-Cost-Prediction-ML-mini-project
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```text
numpy
scipy
scikit-learn
```

## Running the Experiments

Run the scripts from the repository root.

**LASSO regression**

```bash
python src/run_linear.py
```

**Neural network**

```bash
python src/run_nn.py
```

**Random Forest**

```bash
python src/run_random_forest.py
```

**Feature selection and PCA**

```bash
python src/run_feature_selection.py
```

Check `results/model_results.txt` for the recorded experiment results.

## Team Contributions

The project combines reproduction of the paper's core model experiments with an additional model comparison and feature-analysis experiments.

- **LASSO and Random Forest:** regularized linear regression experiments and Random Forest extension.
- **Neural Network and Feature Analysis:** neural network implementation, forward stepwise feature selection, and PCA experiments.

