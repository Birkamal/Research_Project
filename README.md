# Project Introduction
Development of production ready Machine Learning Gradient Boosting Models for Binary Classification.
The project aims to create and compare optimized ML models on multiple data sources for binary classification problem.
This project is done in a team of 5 students under the supervision of Tanaby Zibamanzar Mofrad.

## Project Outline

1. Created **Pre-Processing Module** to clean the data and **Feature Engineering Module** to create new features using a Python framework Featuretools
2. Used **Hyperopt, Optuna** and **Random Search** for hyper-parameter tuning
3. Trained the model using **[XGBoost](https://github.com/Birkamal/Research_Project/blob/master/main_file/mlpipeline/xgb_class.py), Catboost** and **LightGBM** boosting methods
4. Performed **Integration and Unit Testing** using **Pytest** tool of Python
5. Deployed the models on **Google Cloud Platform (GCP)**
6. To check the performance of the models used **Statistical Significance Tests**
7. Comparison of different fitted models and optimization methods against each other and AutoML of GCP
8. Comparison of the results with previous papers

![](https://user-images.githubusercontent.com/56703496/85181382-c8da4980-b253-11ea-8bb4-2e30da00cb7b.png)

## Data Source

The data used are available in public domain:

1. Higgs dataset has been produced using Monte Carlo simulations at Physics & Astronomy, Univ. of California Irvine. The dataset can be found at (http://archive.ics.uci.edu/ml/datasets/HIGGS).

It is a classification problem and identifies exotic particles in high-energy physics based on the sensors information(Signal process produces Higgs bosons (label 1)           and a background process does not (label 0)).

The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features which are high-level features derived by physicists to help discriminate between the two classes. For this project, we ignore the last 7 columns and use Featuretools python library (https://www.featuretools.com/) to create new features and compare with previous studies.

2. Credit card fraud dataset is an imbalanced classification problem ~ 500:1. The dataset is taken from Kaggle has 30 features where 28 are derived from PCA. This problem involves creating the custom evaluation function in LightGBM and XGBoost
