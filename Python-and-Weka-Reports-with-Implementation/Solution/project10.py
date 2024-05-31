#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

data = pd.read_csv('/kaggle/input/heart-dataset/Heart.csv')
data.head()


# In[6]:


print(data.isnull().sum())


# In[7]:


data['PublishedDate'] = pd.to_datetime(data['PublishedDate'])
data['DurationFrom'] = pd.to_datetime(data['DurationFrom'])
data['DurationTo'] = pd.to_datetime(data['DurationTo'])


# ##  Basic Statistical Analysis
# 

# In[8]:


print(data.describe())


# ## Grouping Data

# In[9]:


# Group by period and calculate the sum of deaths
grouped_by_period = data.groupby('Period')[['PersonsDeathsUnder75', 'FemalesDeathsUnder75', 'MalesDeathsUnder75']].sum()
print(grouped_by_period)

# Group by GeoEntityName and calculate the sum of deaths
grouped_by_geoentity = data.groupby('GeoEntityName')[['PersonsDeathsUnder75', 'FemalesDeathsUnder75', 'MalesDeathsUnder75']].sum()
print(grouped_by_geoentity)


# ## Advanced Visualization

# In[10]:


import matplotlib.pyplot as plt

grouped_by_period.plot(kind='bar', stacked=True)
plt.title('Total Deaths Under 75 Over Different Periods')
plt.ylabel('Number of Deaths')
plt.xlabel('Period')
plt.show()


# #### Comparing Deaths by District Council

# In[11]:


import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x='GeoName', y='PersonsDeathsUnder75', data=data)
plt.title('Deaths Under 75 by District Council')
plt.ylabel('Number of Deaths')
plt.xlabel('District Council')
plt.xticks(rotation=45)
plt.show()


# #### Gender Comparison

# In[12]:


gender_data = data.melt(id_vars=['Period', 'GeoName'], value_vars=['FemalesDeathsUnder75', 'MalesDeathsUnder75'], var_name='Gender', value_name='Deaths')

plt.figure(figsize=(12, 8))
sns.barplot(x='GeoName', y='Deaths', hue='Gender', data=gender_data)
plt.title('Comparison of Deaths Under 75 by Gender')
plt.ylabel('Number of Deaths')
plt.xlabel('District Council')
plt.xticks(rotation=45)
plt.show()


# #### Time Series Analysis

# In[13]:


time_series_data = data.groupby(['DurationFrom'])[['PersonsDeathsUnder75', 'FemalesDeathsUnder75', 'MalesDeathsUnder75']].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(time_series_data['DurationFrom'], time_series_data['PersonsDeathsUnder75'], label='Total Deaths')
plt.plot(time_series_data['DurationFrom'], time_series_data['FemalesDeathsUnder75'], label='Female Deaths')
plt.plot(time_series_data['DurationFrom'], time_series_data['MalesDeathsUnder75'], label='Male Deaths')
plt.title('Time Series of Deaths Under 75')
plt.ylabel('Number of Deaths')
plt.xlabel('Time')
plt.legend()
plt.show()


# #### Total Deaths Over Time by GeoEntityName

# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

# Group by DurationFrom and GeoEntityName and calculate the sum of deaths
time_geo_data = data.groupby(['DurationFrom', 'GeoEntityName'])[['PersonsDeathsUnder75']].sum().reset_index()

plt.figure(figsize=(14, 8))
sns.lineplot(data=time_geo_data, x='DurationFrom', y='PersonsDeathsUnder75', hue='GeoEntityName')
plt.title('Total Deaths Over Time by GeoEntityName')
plt.ylabel('Number of Deaths')
plt.xlabel('Year')
plt.legend(title='GeoEntityName')
plt.show()


# #### Heatmap of Deaths by GeoEntityName and Period

# In[15]:


# Pivot the data for the heatmap
heatmap_data = data.pivot_table(index='GeoEntityName', columns='Period', values='PersonsDeathsUnder75', aggfunc='sum')

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Heatmap of Deaths by GeoEntityName and Period')
plt.ylabel('GeoEntityName')
plt.xlabel('Period')
plt.show()


# #### Pie Chart of Total Deaths by Gender
# 

# In[16]:


# Summarize deaths by gender
gender_sums = data[['FemalesDeathsUnder75', 'MalesDeathsUnder75']].sum()

plt.figure(figsize=(8, 8))
plt.pie(gender_sums, labels=['Females', 'Males'], autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
plt.title('Total Deaths by Gender')
plt.show()


# #### Boxplot of Deaths by District Council

# In[17]:


plt.figure(figsize=(14, 8))
sns.boxplot(data=data, x='GeoName', y='PersonsDeathsUnder75')
plt.title('Boxplot of Deaths by District Council')
plt.ylabel('Number of Deaths')
plt.xlabel('District Council')
plt.xticks(rotation=45)
plt.show()


# #### Violin Plot of Deaths by Gender and District Council

# In[18]:


plt.figure(figsize=(14, 8))
sns.violinplot(data=gender_data, x='GeoName', y='Deaths', hue='Gender', split=True)
plt.title('Violin Plot of Deaths by Gender and District Council')
plt.ylabel('Number of Deaths')
plt.xlabel('District Council')
plt.xticks(rotation=45)
plt.show()


# #### Bar Plot of Total Deaths by Period

# In[19]:


plt.figure(figsize=(12, 6))
sns.barplot(data=data, x='Period', y='PersonsDeathsUnder75', estimator=sum, ci=None)
plt.title('Total Deaths by Period')
plt.ylabel('Number of Deaths')
plt.xlabel('Period')
plt.show()


# #### Pair Plot of Deaths by GeoEntityName

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select relevant columns for the pair plot
pairplot_data = data[['GeoName', 'PersonsDeathsUnder75', 'FemalesDeathsUnder75', 'MalesDeathsUnder75']]
pairplot_data = pairplot_data.groupby('GeoName').sum().reset_index()

sns.pairplot(pairplot_data, hue='GeoName', markers="o")
plt.suptitle('Pair Plot of Deaths by GeoEntityName', y=1.02)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




