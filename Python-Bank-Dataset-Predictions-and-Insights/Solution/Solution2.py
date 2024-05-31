#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import warnings
# Ignore warnings
warnings.filterwarnings("ignore")
# Load the dataset
data = pd.read_csv('GoBank.csv')

# Display basic information about the dataset
print(data.info())



# In[73]:


data.isnull().sum()


# In[74]:


print(data.describe())


# In[75]:


print(data.head())


# ## Data Preprocessing and Exploratory Data Analysis (EDA)
# 

# In[76]:


import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fill missing values
data['Qualification'].fillna(data['Qualification'].mode()[0], inplace=True)
data['Last Contact Direction'].fillna(data['Last Contact Direction'].mode()[0], inplace=True)
data['Last Contact Duration'].fillna(data['Last Contact Duration'].mean(), inplace=True)
data['Number of Previous Campaign Calls'].fillna(data['Number of Previous Campaign Calls'].mean(), inplace=True)
data['Previous Campaign Outcome'].fillna(data['Previous Campaign Outcome'].mode()[0], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Qualification', 'Occupation', 'Marital Status', 'Home Mortgage', 'Personal Loan', 'Has Other Bank Account', 'Last Contact Direction', 'Last Contact Month', 'Last Contact Weekday', 'Previous Campaign Outcome', 'Sale Outcome']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature scaling
scaler = StandardScaler()
numerical_columns = ['Age', 'Last Contact Duration', 'Number of Current Campaign Calls', 'Number of Previous Campaign Calls', 'RBA Cash Rate', 'Employment Variation Rate', 'Consumer Confidence Index']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Display the first few rows of the processed dataset
print(data.head())


# In[77]:


data.isnull().sum()


# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('GoBank.csv')

# Fill missing values
data['Qualification'].fillna(data['Qualification'].mode()[0], inplace=True)
data['Last Contact Direction'].fillna(data['Last Contact Direction'].mode()[0], inplace=True)
data['Last Contact Duration'].fillna(data['Last Contact Duration'].mean(), inplace=True)
data['Number of Previous Campaign Calls'].fillna(data['Number of Previous Campaign Calls'].mean(), inplace=True)
data['Previous Campaign Outcome'].fillna(data['Previous Campaign Outcome'].mode()[0], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Qualification', 'Occupation', 'Marital Status', 'Home Mortgage', 'Personal Loan', 'Has Other Bank Account', 'Last Contact Direction', 'Last Contact Month', 'Last Contact Weekday', 'Previous Campaign Outcome', 'Sale Outcome']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature scaling
scaler = StandardScaler()
numerical_columns = ['Age', 'Last Contact Duration', 'Number of Current Campaign Calls', 'Number of Previous Campaign Calls', 'RBA Cash Rate', 'Employment Variation Rate', 'Consumer Confidence Index']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Display the first few rows of the processed dataset
print(data.head())



# # EDA

# ## Correlation Map
# 

# In[79]:


# Exclude CustomerID from the correlation heatmap
data_corr = data.drop(columns=['CustomerID'])

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data_corr.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# ## Visual Analysis of Sale Outcome Factors

# In[80]:


# Sale Outcome distribution
sns.countplot(x='Sale Outcome', data=data)
plt.title('Sale Outcome Distribution')
plt.show()

# Age vs Sale Outcome
sns.boxplot(x='Sale Outcome', y='Age', data=data)
plt.title('Age vs Sale Outcome')
plt.show()

# Qualification vs Sale Outcome
sns.countplot(x='Qualification', hue='Sale Outcome', data=data)
plt.title('Qualification vs Sale Outcome')
plt.show()

# Home Mortgage vs Sale Outcome
sns.countplot(x='Home Mortgage', hue='Sale Outcome', data=data)
plt.title('Home Mortgage vs Sale Outcome')
plt.show()

# Last Contact Direction vs Sale Outcome
sns.countplot(x='Last Contact Direction', hue='Sale Outcome', data=data)
plt.title('Last Contact Direction vs Sale Outcome')
plt.show()

# Previous Campaign Outcome vs Sale Outcome
sns.countplot(x='Previous Campaign Outcome', hue='Sale Outcome', data=data)
plt.title('Previous Campaign Outcome vs Sale Outcome')
plt.show()

# Economic indicators vs Sale Outcome
sns.boxplot(x='Sale Outcome', y='RBA Cash Rate', data=data)
plt.title('RBA Cash Rate vs  Sale Outcome')
plt.show()

sns.boxplot(x='Sale Outcome', y='Employment Variation Rate', data=data)
plt.title('Employment Variation Rate vs Sale Outcome')
plt.show()

sns.boxplot(x='Sale Outcome', y='Consumer Confidence Index', data=data)
plt.title('Consumer Confidence Index vs Sale Outcome')
plt.show()


# The code provides a visual analysis of factors related to the 'Sale Outcome' in a dataset. Here's a breakdown of each visualization and its purpose:
# 
# ## Sale Outcome Distribution:
# 
# This plot shows the distribution of the 'Sale Outcome' variable, indicating the frequency of 'Sale' and 'No Sale' outcomes.
# ## Age vs Sale Outcome:
# 
# This boxplot compares the distribution of ages between customers who made a sale and those who did not, providing insights into how age may influence the sale outcome.
# ## Qualification vs Sale Outcome:
# 
# This countplot displays the distribution of qualification levels among customers, categorized by their sale outcomes, helping to identify any correlation between education level and sale success.
# ## Home Mortgage vs Sale Outcome:
# 
# This countplot illustrates whether customers with a home mortgage are more likely to make a sale compared to those without, aiding in understanding the impact of home mortgages on sale outcomes.
# ## Last Contact Direction vs Sale Outcome:
# 
# This countplot shows how the direction of the last contact (inbound or outbound) correlates with sale outcomes, revealing whether one contact method is more effective than the other.
# ## Previous Campaign Outcome vs Sale Outcome:
# 
# This countplot examines the relationship between the outcome of previous marketing campaigns and the current sale outcome, indicating if past campaign success influences future sales.
# ## RBA Cash Rate vs Sale Outcome:
# 
# This boxplot visualizes how the RBA cash rate (an economic indicator) varies among customers based on their sale outcomes, helping to understand the impact of economic factors on sales.
# ## Employment Variation Rate vs Sale Outcome:
# 
# This boxplot compares the employment variation rate among customers who made a sale versus those who did not, exploring the influence of employment trends on sale success.
# ## Consumer Confidence Index vs Sale Outcome:
# 
# This boxplot shows the distribution of the consumer confidence index among customers, categorized by their sale outcomes, providing insights into the relationship between consumer sentiment and sales.
# Overall, these visualizations offer a comprehensive understanding of how various demographic, economic, and campaign-related factors contribute to sales outcomes in the dataset.

# ## Visual Analysis of Sale Outcome Factors
# 
# This below code sets up a matplotlib figure with multiple subplots to provide a visual analysis of factors related to the 'Sale Outcome' in a dataset. Each subplot represents a different analysis, including the distribution of sale outcomes, the relationship between age and sale outcomes, the influence of qualification and home mortgage on sale outcomes, the impact of the last contact direction and previous campaign outcomes, and the association between economic indicators (RBA Cash Rate, Employment Variation Rate, Consumer Confidence Index) and sale outcomes. The plots facilitate easy comparison and interpretation of how various demographic, campaign-related, and economic factors affect the likelihood of a successful sale.

# In[81]:


# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 10))

# Sale Outcome distribution
sns.countplot(x='Sale Outcome', data=data, ax=axes[0, 0])
axes[0, 0].set_title('Sale Outcome Distribution')

# Age vs Sale Outcome
sns.boxplot(x='Sale Outcome', y='Age', data=data, ax=axes[0, 1])
axes[0, 1].set_title('Age vs Sale Outcome')

# Qualification vs Sale Outcome
sns.countplot(x='Qualification', hue='Sale Outcome', data=data, ax=axes[1, 0])
axes[1, 0].set_title('Qualification vs Sale Outcome')

# Home Mortgage vs Sale Outcome
sns.countplot(x='Home Mortgage', hue='Sale Outcome', data=data, ax=axes[1, 1])
axes[1, 1].set_title('Home Mortgage vs Sale Outcome')

# Last Contact Direction vs Sale Outcome
sns.countplot(x='Last Contact Direction', hue='Sale Outcome', data=data, ax=axes[2, 0])
axes[2, 0].set_title('Last Contact Direction vs Sale Outcome')

# Previous Campaign Outcome vs Sale Outcome
sns.countplot(x='Previous Campaign Outcome', hue='Sale Outcome', data=data, ax=axes[2, 1])
axes[2, 1].set_title('Previous Campaign Outcome vs Sale Outcome')

# RBA Cash Rate vs Sale Outcome
sns.boxplot(x='Sale Outcome', y='RBA Cash Rate', data=data, ax=axes[3, 0])
axes[3, 0].set_title('RBA Cash Rate vs Sale Outcome')

# Employment Variation Rate vs Sale Outcome
sns.boxplot(x='Sale Outcome', y='Employment Variation Rate', data=data, ax=axes[3, 1])
axes[3, 1].set_title('Employment Variation Rate vs Sale Outcome')

# Consumer Confidence Index vs Sale Outcome
sns.boxplot(x='Sale Outcome', y='Consumer Confidence Index', data=data, ax=axes[4, 0])
axes[4, 0].set_title('Consumer Confidence Index vs Sale Outcome')

# Hide the last unused subplot (axes[4, 1])
fig.delaxes(axes[4, 1])

plt.tight_layout()
plt.show()


# # Questions to address from GOBank.csv Dataset

# In[82]:


# 1. Demographic details influence on sales outcomes
# Age vs Sale Outcome
sns.boxplot(x='Sale Outcome', y='Age', data=data)
plt.title('Age vs Sale Outcome')
plt.show()

# Qualification vs Sale Outcome
sns.countplot(x='Qualification', hue='Sale Outcome', data=data)
plt.title('Qualification vs Sale Outcome')
plt.xticks(rotation=45)
plt.show()



# In[83]:


# 2. Influence of having different types of accounts on Sale Outcome
# Home Mortgage vs Sale Outcome
sns.countplot(x='Home Mortgage', hue='Sale Outcome', data=data)
plt.title('Home Mortgage vs Sale Outcome')
plt.show()

# Personal Loan vs Sale Outcome
sns.countplot(x='Personal Loan', hue='Sale Outcome', data=data)
plt.title('Personal Loan vs Sale Outcome')
plt.show()

# Has Other Bank Account vs Sale Outcome
sns.countplot(x='Has Other Bank Account', hue='Sale Outcome', data=data)
plt.title('Has Other Bank Account vs Sale Outcome')
plt.show()




# In[84]:


# 3. Influence of last contact method on Sale Outcome
sns.countplot(x='Last Contact Direction', hue='Sale Outcome', data=data)
plt.title('Last Contact Direction vs Sale Outcome')
plt.show()



# In[85]:


# 4. Influence of previous campaign outcome on Sale Outcome
sns.countplot(x='Previous Campaign Outcome', hue='Sale Outcome', data=data)
plt.title('Previous Campaign Outcome vs Sale Outcome')
plt.show()



# In[86]:


# 5. Impact of economic indicators on Sale Outcomes
# RBA Cash Rate vs Sale Outcome
sns.boxplot(x='Sale Outcome', y='RBA Cash Rate', data=data)
plt.title('RBA Cash Rate vs Sale Outcome')
plt.show()

# Employment Variation Rate vs Sale Outcome
sns.boxplot(x='Sale Outcome', y='Employment Variation Rate', data=data)
plt.title('Employment Variation Rate vs Sale Outcome')
plt.show()

# Consumer Confidence Index vs Sale Outcome
sns.boxplot(x='Sale Outcome', y='Consumer Confidence Index', data=data)
plt.title('Consumer Confidence Index vs Sale Outcome')
plt.show()

# 6. Exploration of other factors affecting sales outcomes
# Additional analysis can be performed here based on the specific business context and available data.


# In[ ]:





# ## Predictive Machine Learning Model
# We'll use Logistic Regression to predict 'Sale' or 'No Sale'.

# In[87]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split the data into features and target variable
X = data.drop(columns=['CustomerID', 'Sale Outcome'])
y = data['Sale Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Logistics Regression Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')


# ## Clustering Model
# We'll use K-Means Clustering to identify customer segments.

# In[88]:


from sklearn.cluster import KMeans

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Train the K-Means model with the optimal number of clusters
optimal_clusters = 3  # Assume 3 for this example
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
sns.scatterplot(x='Age', y='Last Contact Duration', hue='Cluster', data=data, palette='viridis')
plt.title('Customer Segments')
plt.show()


# In[ ]:




