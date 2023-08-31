#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pandas


# In[2]:


#pip install matplotlib


# In[3]:


#pip install statsmodels


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# In[5]:


def cleanUpData(df, colm_name):
    # Calculate Q1, Q2 and IQR
    Q1 = df[colm_name].quantile(0.25)
    Q3 = df[colm_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define the acceptable range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the data
    df = df[(df[colm_name] >= lower_bound) & (df[colm_name] <= upper_bound)]


# In[6]:


# Load the data
df = pd.read_csv('/Users/omidheravi/Downloads/datavals.csv')

# Strip leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Difference'] = df['CurrentBarClose'] - df['PreviousBarClose']

df = df[df['POCDiff'] >= -100]

# Names of all the columns
print(df.columns)


# In[7]:


df = df.dropna()


# In[8]:


# Now we can plot each column to get a sense of the data
for col in df.columns:
    df[col].plot()
    plt.show()


# In[9]:


print(df['POCDiff'].describe())


# In[10]:


cleanUpData(df, 'POCDiff');


# In[11]:


df['POCDiff'].hist(bins=10)
plt.xlabel('POCDiff')
plt.ylabel('Frequency')
plt.show()


# In[12]:


df.boxplot(column=['POCDiff'])
plt.show()


# In[13]:


#print(df['Difference'].describe())
#print(df['Difference'].describe())
#print(df['Difference'].describe())
#print(df['Difference'].describe())
#print(df['Difference'].describe())


# In[14]:


cleanUpData(df, 'Difference');


# In[15]:


df['Difference'].hist(bins=10)
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.show()


# In[16]:


df.boxplot(column=['Difference'])
plt.show()


# In[17]:


#print(df.corr()['Difference'])
#print(df.corr()['EntropyValue'])
#print(df.corr()['EntropyValue'])
#print(df.corr()['EntropyValue'])


# In[18]:


#print(df.corr()['EntropyValue'])


# In[21]:


# Calculate the rolling mean and standard deviation
df['Rolling Mean'] = df['Difference'].rolling(window=12).mean()
df['Rolling Std'] = df['Difference'].rolling(window=12).std()


# In[22]:


# Plot the rolling mean and standard deviation
df[['Difference', 'Rolling Mean', 'Rolling Std']].plot()
plt.show()


# In[28]:


# Descriptive Statistics
print(df.describe())


# In[27]:


# Check for Missing Values
#print(df.isnull().sum())


# In[30]:


print(df.corr())


# In[44]:


# Define a threshold for high POCDiff and BuySellRate
poc_diff_threshold = df['POCDiff'].quantile(0.95)  # 90th percentile
buy_sell_rate_threshold = df['BuySellRate'].quantile(0.655)  # 90th percentile

# Generate potential sell signals based on POCDiff
#df['HighProbPOCDiff'] = df['POCDiff'] > poc_diff_threshold

# Generate potential buy signals based on BuySellRate
#df['HighProbVolRate'] = df['BuySellRate'] > buy_sell_rate_threshold


# In[45]:


print(poc_diff_threshold)


# In[46]:


print(buy_sell_rate_threshold)


# In[ ]:




