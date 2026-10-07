# Metabolic Cost Prediction — ML Mini Project

Reproduction and extension of the paper:

**"Predicting Metabolic Cost During Human-in-the-Loop Optimization"**

This project reproduces the main machine learning methodology described in the reference paper and extends it with an additional machine learning model.

## Project Overview

The goal of this project is to predict human metabolic cost from biomechanical and EMG-related features collected during human-in-the-loop optimization experiments.

The dataset contains measurements from two subjects, with 29 input features and a metabolic-cost target.

The project focuses on:

1. Reproducing the machine learning approaches described in the reference paper.
2. Evaluating different feature configurations.
3. Performing feature selection and dimensionality reduction.
4. Extending the work with an additional machine learning model.

## Reference Paper

**Predicting Metabolic Cost During Human-in-the-Loop Optimization**  
Eley Ng and Erez Krimsky, CS229, 2018.

Reference paper:

https://cs229.stanford.edu/proj2018/report/11.pdf

## Models

### LASSO Linear Regression

- L1-regularized linear regression
- 10-fold cross-validation for regularization selection
- Evaluated using Mean Squared Error (MSE)

### Neural Network

- Single hidden-layer neural network
- `tanh` activation
- Linear output
- Cross-validation to select the number of hidden neurons

### Forward Stepwise Feature Selection

Features are selected iteratively based on cross-validation performance.

### PCA

Principal Component Analysis is used for dimensionality reduction after feature normalization.

### Random Forest Extension

Random Forest Regression is included as the additional machine learning model for the project extension.

The Random Forest model is evaluated using cross-validation and compared with the reproduced models.

## Feature Configurations

The models are evaluated using four feature configurations:

- **All features**
- **Step + EMG**
- **Step only**
- **EMG only**

The 29 input features include:

- Force/step-related features
- Step timing and width
- EMG features
- Control features

## Dataset

The processed dataset contains data from two subjects:

- Subject 1
- Subject 2

Each subject contains approximately 180 samples and 29 input features.

The metabolic-cost measurement is used as the prediction target.

The processed dataset is stored in:

```text
data/processed_data.mat
```

## Preprocessing

The preprocessing pipeline includes:

- Feature standardization
- Removal of samples with negative metabolic-cost targets where required
- Separation into the four feature configurations
- Cross-validation for model evaluation and selection

## Evaluation

Model performance is primarily evaluated using:

**Mean Squared Error (MSE)**

Cross-validation is used for model selection and performance estimation.

For the neural network, cross-validation is used to investigate different hidden-layer sizes.

For LASSO regression, cross-validation is used to select the regularization parameter.

## Results

The experimental results for all models are available in:

```text
results/model_results.txt
```

The results include:

- LASSO Linear Regression
- Neural Network
- Forward Stepwise Feature Selection
- PCA
- Random Forest Extension

## Project Structure

```text
Metabolic-Cost-Prediction-ML-mini-project/
│
├── data/
│   └── processed_data.mat
│
├── figures/
│
├── results/
│   └── model_results.txt
│
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
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies Used

- **Python 3**
- **NumPy** — numerical computation
- **SciPy** — loading MATLAB `.mat` data
- **scikit-learn** — machine learning models and cross-validation
- **Git / GitHub** — version control and collaboration
- **VS Code** — development environment

## Reproducibility

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the individual experiments from the project root:

```bash
python src/run_linear.py
```

```bash
python src/run_nn.py
```

```bash
python src/run_feature_selection.py
```

```bash
python src/run_random_forest.py
```

The resulting model performance can be compared using the MSE values reported in:

```text
results/model_results.txt
```

## Team Contributions

The project was developed collaboratively using separate Git branches.

- **Person 1:** LASSO Linear Regression and Random Forest extension
- **Person 2:** Neural Network reproduction, forward stepwise feature selection, and PCA

The implementations were integrated into the `main` branch after review.

## Academic Project

This repository was developed as part of an ML mini-project based on the referenced CS229 paper.
