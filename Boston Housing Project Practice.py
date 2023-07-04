#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[22]:


import pandas as pd
df = pd.read_csv('BostonHousing.csv')
df = df.fillna(df.mean())


# In[4]:


# Step 1: Load the dataset
housing_df = pd.read_csv('BostonHousing.csv')


# In[16]:


housing_df.shape


# In[10]:


housing_df = housing_df.rename(columns={'CAT. MEDV': 'CAT_MEDV'})
housing_df.head(9)


# In[7]:


print(housing_df) 


# In[11]:


housing_df.columns = [s.strip().replace('', '_') for s in housing_df.columns]


# In[12]:


housing_df.loc[0:3] 


# In[15]:


pd.concat([housing_df.iloc[4:6,0:2], housing_df.iloc[4:6,4:6]], axis=1)


# In[16]:


housing_df.columns


# In[5]:


# Further Data Exploration
print(housing_df.describe()) 
housing_df.hist(figsize=(12, 10))  


# In[6]:


# Data Preprocessing
# Handle missing values (if any)
# For example, to drop rows with missing values:
housing_df.dropna(inplace=True)


# In[7]:


# Handle outliers (if any)
# For example, to remove outliers using z-score:
z_scores = (housing_df - housing_df.mean()) / housing_df.std()
housing_df = housing_df[(z_scores < 3).all(axis=1)]


# In[8]:


# Step 5: Splitting the Data
X = housing_df.drop('MEDV', axis=1)  # Features
y = housing_df['MEDV']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


# Step 6: Model Selection and Training
model = LinearRegression()
model.fit(X_train, y_train)


# In[10]:


# Step 7: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[15]:


import matplotlib.pyplot as plt


# In[13]:


# Pandas Version
ax = housing_df.MEDV.hist()
ax.set_xlabel('MEDV'); ax.set_ylabel('count');


# In[18]:


# Pandas Version
# boxplot of MEDV for different values of CHAS
ax = housing_df.boxplot(column='MEDV', by='CHAS')
ax.set_xlabel('MEDV'); ax.set_ylabel('count')
plt.suptitle('') # Suppress the titles
plt.title('')


# In[17]:


# Using seaborn 
# Simple heatmap of correlations (without values)
import seaborn as sns
corr = housing_df.corr()
sns.heatmap(corr)


# In[22]:


# Change to divergent scale and fix the range
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, 
vmax=1, cmap="RdBu")


# In[25]:


import pandas as pd
import matplotlib.pyplot as plt

housing_df = pd.read_csv('BostonHousing.csv')

fig, ax = plt.subplots()
ax.hist(housing_df.MEDV)
ax.set_axisbelow(True)  # Show the grid lines behind the histogram
ax.grid(which='major', color='grey', linestyle='--')
plt.suptitle('')  # Suppress the titles
plt.title('')

plt.show()  # Display the histogram


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the housing data into a DataFrame
housing_df = pd.read_csv('BostonHousing.csv')

# Adding Variables
# Color the points by the value of MEDV
housing_df.plot.scatter(x='LSTAT', y='NOX', c=['C0' if c == 1 else 'C1' for c in housing_df.MEDV])

# Show the scatter plot
plt.show()


# In[15]:


## scatter plot: regular and log scale
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
# regular scale
housing_df.plot.scatter(x='CRIM', y='MEDV', ax=axes[0])
# log scale
ax = housing_df.plot.scatter(x='CRIM', y='MEDV', logx=True, logy=True, ax=axes[1])
ax.set_yticks([5, 10, 20, 50])
ax.set_yticklabels([5, 10, 20, 50])
plt.show()


# In[ ]:




