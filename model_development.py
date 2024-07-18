import pandas as pd

# Load the dataset
file_path = 'churn-bigml-80.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder

# Convert categorical variables to numerical format
label_encoder = LabelEncoder()

data['State'] = label_encoder.fit_transform(data['State'])
data['International plan'] = label_encoder.fit_transform(data['International plan'])
data['Voice mail plan'] = label_encoder.fit_transform(data['Voice mail plan'])
data['Churn'] = label_encoder.fit_transform(data['Churn'])

# Display the first few rows of the transformed dataset
print(data.head())


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Scale the features (excluding the target variable 'Churn')
features = data.drop('Churn', axis=1)
scaled_features = scaler.fit_transform(features)

# Convert the scaled features back to a DataFrame
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

# Add the target variable 'Churn' back to the DataFrame
scaled_features_df['Churn'] = data['Churn']

# Display the first few rows of the scaled dataset
print(scaled_features_df.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of target variable 'Churn'
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=scaled_features_df)
plt.title('Distribution of Target Variable (Churn)')
plt.show()

# Correlation matrix
plt.figure(figsize=(14, 10))
correlation_matrix = scaled_features_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot of a subset of features to visualize relationships
subset_features = scaled_features_df[['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes', 'Churn']]
sns.pairplot(subset_features, hue='Churn', diag_kind='kde')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of target variable 'Churn'
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=scaled_features_df)
plt.title('Distribution of Target Variable (Churn)')
plt.show()

# Correlation matrix
plt.figure(figsize=(14, 10))
correlation_matrix = scaled_features_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot of a subset of features to visualize relationships
subset_features = scaled_features_df[['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes', 'Churn']]
sns.pairplot(subset_features, hue='Churn', diag_kind='kde')
plt.show()
from sklearn.model_selection import train_test_split

# Define features and target variable
X = scaled_features_df.drop('Churn', axis=1)
y = scaled_features_df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Define models
logistic_model = LogisticRegression(random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)
gradient_boosting_model = GradientBoostingClassifier(random_state=42)

# Define hyperparameters for tuning
logistic_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

random_forest_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

gradient_boosting_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Hyperparameter tuning using GridSearchCV
logistic_grid = GridSearchCV(logistic_model, logistic_params, cv=5, scoring='accuracy')
random_forest_grid = GridSearchCV(random_forest_model, random_forest_params, cv=5, scoring='accuracy')
gradient_boosting_grid = GridSearchCV(gradient_boosting_model, gradient_boosting_params, cv=5, scoring='accuracy')

# Fit the models
logistic_grid.fit(X_train, y_train)
random_forest_grid.fit(X_train, y_train)
gradient_boosting_grid.fit(X_train, y_train)

# Get the best models
best_logistic_model = logistic_grid.best_estimator_
best_random_forest_model = random_forest_grid.best_estimator_
best_gradient_boosting_model = gradient_boosting_grid.best_estimator_
print("model dump")

from joblib import dump

# Save the best models
dump(best_logistic_model, 'best_logistic_model.joblib')
dump(best_random_forest_model, 'best_random_forest_model.joblib')
dump(best_gradient_boosting_model, 'best_gradient_boosting_model.joblib')

print("Models saved successfully!")

print("Predictions starting ")
logistic_preds = best_logistic_model.predict(X_test)
random_forest_preds = best_random_forest_model.predict(X_test)
gradient_boosting_preds = best_gradient_boosting_model.predict(X_test)
print("Predictions ending")
# Evaluate the models
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix
print("statring ")
logistic_metrics = evaluate_model(y_test, logistic_preds)
random_forest_metrics = evaluate_model(y_test, random_forest_preds)
gradient_boosting_metrics = evaluate_model(y_test, gradient_boosting_preds)
print("ending")
(logistic_metrics, random_forest_metrics, gradient_boosting_metrics)
# Display model evaluation results
def print_metrics(model_name, metrics):
    accuracy, precision, recall, f1, roc_auc, conf_matrix = metrics
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\n")

print_metrics("Logistic Regression", logistic_metrics)
print_metrics("Random Forest", random_forest_metrics)
print_metrics("Gradient Boosting", gradient_boosting_metrics)
# Plot comparison graph
def plot_comparison(metrics_dict):
    df = pd.DataFrame(metrics_dict).T
    df.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    df.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Metrics')
    plt.xticks(rotation=0)
    plt.legend(loc='best')
    plt.show()

metrics_dict = {
    'Logistic Regression': logistic_metrics[:5],
    'Random Forest': random_forest_metrics[:5],
    'Gradient Boosting': gradient_boosting_metrics[:5]
}

plot_comparison(metrics_dict)

def plot_comparison_line(metrics_dict):
    df = pd.DataFrame(metrics_dict).T
    df.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    df.plot(kind='line', marker='o', figsize=(12, 8))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Metrics')
    plt.xticks(range(len(df.columns)), df.columns, rotation=45)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

metrics_dict = {
    'Logistic Regression': logistic_metrics[:5],
    'Random Forest': random_forest_metrics[:5],
    'Gradient Boosting': gradient_boosting_metrics[:5]
}

plot_comparison_line(metrics_dict)

import numpy as np

def plot_comparison_radar(metrics_dict):
    df = pd.DataFrame(metrics_dict).T
    df.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

    # Number of variables
    categories = list(df.columns)
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variables)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)

    # Plot each model
    for model, metrics in df.iterrows():
        values = metrics.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.4)

    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title('Model Comparison Radar Plot')
    plt.show()

metrics_dict = {
    'Logistic Regression': logistic_metrics[:5],
    'Random Forest': random_forest_metrics[:5],
    'Gradient Boosting': gradient_boosting_metrics[:5]
}

plot_comparison_radar(metrics_dict)
