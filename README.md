"# AtulPokharelChurnPredictionTask" 





```
### your program is  ready
## contact for more details
```
```bash
atulpokharel12@gmail.com
```



# Customer Churn Prediction

This repository contains code and documentation for predicting customer churn using machine learning models. The dataset used in this project is `churn-bigml-80.csv`. The process involves data preprocessing, model training, and evaluation of three different models: Logistic Regression, Random Forest, and Gradient Boosting.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Insights and Observations](#insights-and-observations)

## Dataset

The dataset used in this project is `churn-bigml-80.csv`. It contains various features related to customer churn, including customer demographics, account information, and usage patterns.

## Installation

To run the code in this repository, you need to have Python installed along with the required libraries. You can install the necessary dependencies using pip:

To run this project you would need:

- Download/ Clone the project

```git
  https://github.com/atulpokharel-gp/AtulPokharelChurnPredictionTask-.git
```

- Create a virtual environment

```python3
  python3 -m venv env
```

- Activate the environment
```bash
  source env/bin/activate
```
- for linux
```bash
./env/Scripts/activate
```
- for window
 Install the required packages

```
  python3
  pip3 install -r requirements.txt
```
``` 
 install cudn according to tensorflow-gpu version
```

- Run the project
```python3
  python model_development.py

```

# Data Preprocessing
The data preprocessing steps include:
```
Loading the dataset.
Checking for missing values.
Encoding categorical variables to numerical format.
Scaling the features using StandardScaler.
```
# Model Training and Evaluation
The models used in this project are:
```
Logistic Regression
Random Forest
Gradient Boosting
We perform hyperparameter tuning using GridSearchCV and evaluate the models based on accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.
```

# Insights and Observations

```
Based on the evaluation metrics, the model performances can be compared to determine which model is the best for predicting customer churn. 
Further analysis and hyperparameter tuning can be performed to improve the models' performance.
```