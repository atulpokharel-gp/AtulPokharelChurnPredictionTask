import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from tabulate import tabulate
import plotext as plt_ext
import seaborn as sns
import os 

# Load the dataset
file_path = 'churn-bigml-80.csv'
data = pd.read_csv(file_path)
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("missing_values",missing_values)
# Encode categorical variables
label_enc = LabelEncoder()
data['State'] = label_enc.fit_transform(data['State'])
data['International plan'] = label_enc.fit_transform(data['International plan'])
data['Voice mail plan'] = label_enc.fit_transform(data['Voice mail plan'])
data['Churn'] = label_enc.fit_transform(data['Churn'])
print("LabelEncoder",data)
# Scale the features
scaler = StandardScaler()
numerical_features = data.drop(columns=['Churn']).columns
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print("Scale_features",data)



# Create a folder named 'histograms' if it doesn't exist
if not os.path.exists('histograms'):
    os.makedirs('histograms')

# Descriptive statistics
descriptive_stats = data.describe()

# Save descriptive statistics to a text file
with open('descriptive_statistics.txt', 'w') as f:
    f.write(tabulate(descriptive_stats, headers='keys', tablefmt='psql'))

# Plot distribution of numerical features
numerical_features = data.drop(columns=['Churn']).columns
fig, axes = plt.subplots((len(numerical_features) + 2) // 3, 3, figsize=(15, 20))
fig.tight_layout(pad=5.0)

for i, col in enumerate(numerical_features):
    sns.histplot(data[col], ax=axes[i//3, i%3], kde=True)
    axes[i//3, i%3].set_title(f'Distribution of {col}')

    # Save individual histograms
    plt.figure()
    plt.hist(data[col], bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f'histograms/{col}_distribution.png')

# Remove empty subplots
for j in range(i+1, (len(numerical_features) + 2) // 3 * 3):
    fig.delaxes(axes[j//3, j%3])

plt.show()

# Correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualize the relationship between some key features and the target variable
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.tight_layout(pad=5.0)

sns.boxplot(x='Churn', y='Total day minutes', data=data, ax=axes[0, 0])
axes[0, 0].set_title('Total Day Minutes vs Churn')

sns.boxplot(x='Churn', y='Total eve minutes', data=data, ax=axes[0, 1])
axes[0, 1].set_title('Total Eve Minutes vs Churn')

sns.boxplot(x='Churn', y='Total night minutes', data=data, ax=axes[0, 2])
axes[0, 2].set_title('Total Night Minutes vs Churn')

sns.boxplot(x='Churn', y='Total intl minutes', data=data, ax=axes[1, 0])
axes[1, 0].set_title('Total Intl Minutes vs Churn')

sns.boxplot(x='Churn', y='Customer service calls', data=data, ax=axes[1, 1])
axes[1, 1].set_title('Customer Service Calls vs Churn')

plt.show()

data.to_csv("clean.csv", index=False)