# Metabolic Cost Prediction — ML Mini Project

Reproduction and extension of the paper:

**"Predicting Metabolic Cost During Human-in-the-Loop Optimization"**

This project reproduces the machine learning methods described in the reference paper and extends the work with an additional machine learning model.

## Project Overview

The goal of this project is to predict human metabolic cost from biomechanical and EMG-related features collected during human-in-the-loop optimization experiments.

The original dataset contains measurements from two subjects, with 29 input features and a metabolic-cost target.

Our project focuses on:

1. Reproducing the machine learning approaches described in the reference paper.
2. Evaluating different feature configurations.
3. Performing feature selection and dimensionality reduction.
4. Extending the original work with an additional machine learning model.

## Reference Paper

**Predicting Metabolic Cost During Human-in-the-Loop Optimization**  
Eley Ng and Erez Krimsky, CS229, 2018.

Reference:
https://cs229.stanford.edu/proj2018/report/11.pdf

## Models

### Reproduction Models

The project implements the following approaches from the reference methodology:

- **LASSO Linear Regression**
  - L1-regularized linear regression
  - 10-fold cross-validation for regularization selection

- **Neural Network**
  - Single hidden-layer architecture
  - `tanh` activation
  - Cross-validation to select the number of hidden neurons

- **Forward Stepwise Feature Selection**
  - Iteratively selects features based on cross-validation performance

- **PCA**
  - Dimensionality reduction after feature normalization
  - Components are selected based on explained variance

### Extension Model

- **Random Forest Regression**
  - Added as an additional model beyond the approaches used in the reference paper
  - Evaluated using cross-validation
  - Results are compared with the reproduction models

## Feature Configurations

The models are evaluated using four feature configurations:

- **All features**
- **Step + EMG**
- **Step only**
- **EMG only**

The original dataset contains 29 features, including:

- Force/step-related features
- Step timing and width
- EMG features
- Control features

## Dataset

The project uses the processed dataset provided with the reference project.

The processed data contains measurements for two subjects:

- Subject 1
- Subject 2

Each subject contains approximately 180 samples and 29 input features.

The metabolic-cost measurements are used as the prediction target.

## Preprocessing

The preprocessing pipeline includes:

- Feature standardization
- Removal of samples with negative metabolic-cost targets where required
- Separation of the four feature configurations
- Cross-validation during model evaluation

## Evaluation

Model performance is primarily evaluated using **Mean Squared Error (MSE)**.

Cross-validation is used for model selection and performance estimation.

For the neural network, cross-validation is used to investigate different hidden-layer sizes.

For LASSO regression, cross-validation is used to select the regularization parameter.

## Extension Results

The Random Forest model was evaluated on both subjects and all four feature configurations.

Results are stored in:

`results/model_results.txt`

The file contains results for:

- LASSO Linear Regression
- Neural Network
- Feature Selection / PCA
- Random Forest

## Technologies Used

- **Python 3**
- **NumPy** — numerical computation
- **SciPy** — loading MATLAB `.mat` data
- **scikit-learn** — machine learning models and cross-validation
- **Jupyter Notebook** — experimentation and analysis
- **Git / GitHub** — version control and collaboration
- **VS Code** — development environment

## Project Organization

The repository contains separate areas for:

- Dataset files
- Source code
- Experiments and notebooks
- Results
- Figures

The exact final folder structure may be updated as the project is combined between team members.

## Reproducibility

The experiments can be reproduced using the Python scripts in the `src/` directory.

Example:

```bash
python src/run_linear.py
