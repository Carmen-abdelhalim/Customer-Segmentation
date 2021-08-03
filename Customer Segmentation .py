#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation
# ### Introduction
# this is a real dataset containing anonymized customer transactions from an online retailer.
# 
# >I'll apply practical customer behavioral analytics and segmentation techniques.
# 
# >first I will build easy to interpret customer segments. On top of that, I will prepare the segments I'll create, making them ready for machine learning.
# 
# >Finally I will make segments more powerful with k-means clustering to identify similar groups of customers based on their purchasing behavior

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 


# In[2]:


online = pd.read_csv('online.csv')
online.head()


# ## Data Prepration

# In[3]:


# convert date column from string to datetime
online['InvoiceDate']= pd.to_datetime(online['InvoiceDate'])
online.info()


# In[4]:


# calculate TotalSum column by multiply quantity and unit price
online['TotalSum'] = online['Quantity'] * online['UnitPrice']
online.head()


# In[5]:


print('Min:{}; Max:{}'.format(min(online.InvoiceDate),max(online.InvoiceDate)))


# In[6]:


# created a hypothetical snapshot_date variable that can use it to calculate recency
snapshot_date = max(online.InvoiceDate) + datetime.timedelta(days=1)
snapshot_date


# # Calculate RFM Metrics

# ## RFM Segmentation
# 
# I will create customer segments based on Recency, Frequency, Monetary Value analysis by calculating three customer behavior matrics
# - **Recency (R)** how recent was each customer's last purchase.
# - **Frequency (F)** which measures number of "transactions" how many purchases the customer has done in last 12 months.
# - **Monetary Value (M)** measures how much customer spent in the last 12 months.
# 
# Use these values to assign customers to RFM segments then group them into sort of categorization such as high, medium and low.

# In[7]:


# Calculate Recency, Frequency and Monetary value for each customer 

# Aggregate data on a customer level
datamart = online.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

# Rename the columns 
datamart.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalSum': 'MonetaryValue'}, inplace=True)

datamart.head()


# In[8]:


datamart.info()


# ## Building RFM Segments 
# > I will now pre processing data to build kmeans model to segment the customers into three separate groups based on Recency,Frequency and Monetary Value

# ## Data pre-processing for clustering
# 

# In[9]:


# Calculate statistics of variables
datamart.describe()


#  as we can see, the averages and standard deviations are different across the variables.

# ## Exploring distribution

# In[10]:


# exploring distributions of RFM variables!
plt.figure(figsize= [6,8])
plt.subplot(3, 1, 1)
sns.distplot(datamart['Recency'])

plt.subplot(3, 1, 2)
sns.distplot( datamart['Frequency'])

plt.subplot(3, 1, 3)
sns.distplot( datamart['MonetaryValue'])


# - As we can see, Recency metric has a tail on the right, which is mean the Recency is skewed
# 
# - frequency aslo have right skewed, the majority of observations are between zero and 100, while there are values spreading up to fourteen hundered
# 
# - MonetaryValue also have right skewed, the majority of observations are between zero and 800.

# In[11]:


datamart.info()


# ##  unskewed and normalized the dataset

# In[12]:


# unskew data
datamart_log = np.log(datamart)

#Normalize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(datamart_log)

#transform and store the data by scaling and centering it with scaler.
datamart_normalized = scaler.transform(datamart_log)

# Create a pandas DataFrame
datamart_normalized = pd.DataFrame(data=datamart_normalized, index=datamart.index, columns=datamart.columns)


# In[13]:


datamart_normalized


# ## Visualize the normalized variables
# 

# In[14]:


plt.figure(figsize= [6,8])


# Plot recency distribution
plt.subplot(3, 1, 1); sns.distplot(datamart_normalized['Recency'])

# Plot frequency distribution
plt.subplot(3, 1, 2); sns.distplot(datamart_normalized['Frequency'])

# Plot monetary value distribution
plt.subplot(3, 1, 3); sns.distplot(datamart_normalized['MonetaryValue'])

# Show the plot
plt.show()


# **As we can see, the skewness is managed after applying  transformations**

#  # Run K-Means Model 
#  >to identify customer clusters based on their recency, frequency, and monetary value.

# ### 1- define the best number of cluster using elbow criterion method
# 

# In[15]:


# Calculate sum of squared errors

# Import KMeans 
from sklearn.cluster import KMeans

# initialized an empty dictionary to store sum of squared errors 
sse = {}
# Fit KMeans and calculate SSE for each k
for k in range(1, 21):
  
    # Initialize KMeans with k clusters
    kmeans = KMeans(n_clusters= k, random_state=1)
    
    # Fit KMeans on the normalized dataset
    kmeans.fit(datamart_normalized)
    
    # Assign sum of squared distances to k element of dictionary
    sse[k] = kmeans.inertia_


# ### Plot sum of squared errors for each value of k to identify if there is an elbow

# In[16]:


# Add the plot title "The Elbow Method"
plt.title('The Elbow Method')

# Add X-axis label "k"
plt.xlabel('k')

# Add Y-axis label "SSE"
plt.ylabel('SSE')

# Plot SSE values for each key in the dictionary
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()


# **we can see the elbow is clearly around 3-4 clusters** so I am going to use 3 for clusters to segment data

# In[17]:


# Initialize KMeans
kmeans = KMeans(n_clusters=3, random_state=1) 

# Fit k-means clustering on the normalized data set
kmeans.fit(datamart_normalized)

# Extract cluster labels
cluster_labels = kmeans.labels_


# ## Analyizing & Visualizing segments
# > using summary statistics of each cluster and snake plots to understand and compare segments

# In[18]:


# Summary statistics of each cluster 
# Create a DataFrame by adding a new cluster label column
datamart_rfm_k3 = datamart.assign(Cluster=cluster_labels)

# Group the data by cluster
grouped = datamart_rfm_k3.groupby(['Cluster'])

# Calculate average RFM values and segment sizes per cluster value
grouped.agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
  }).round(1)


# we can see the differences in RFM values of these segments

# ### Visualize snake plot
# 

# In[19]:


# created a melted DataFrame to build a snake plot
# Melt the normalized dataset and reset the index
datamart_melt = pd.melt(datamart_rfm_k3.reset_index(), 
                        
# Assign CustomerID and Cluster as ID variables
                    id_vars =['CustomerID', 'Cluster'],

# Assign RFM values as value variables
                    value_vars =['Recency', 'Frequency', 'MonetaryValue'], 
                        
# Name the variable and value
                    var_name='Metric', value_name='Value')
datamart_melt.head()


# In[20]:


# use the melted dataset to build the snake plot.

plt.figure(figsize=[8,6])
# Add the plot title
plt.title('Snake plot of normalized variables')

# Add the x axis label
plt.xlabel('Metric')

# Add the y axis label
plt.ylabel('Value')

# Plot a line for each value of the cluster variable
sns.lineplot(data=datamart_melt, x='Metric', y='Value', hue='Cluster')
plt.show()


# In[ ]:





# In[ ]:




