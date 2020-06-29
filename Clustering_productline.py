#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans


# In[2]:


# reading the data and looking at the first five rows of the data
data=pd.read_csv("Project_clusterings.csv")
data.head()


# In[4]:


# standardizing the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['C','G','Ke','Ko','N','O','T','V','unique_customers','sales','units']])

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# In[12]:


#defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=5, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)


# In[13]:


pred=kmeans.predict(data_scaled)


# In[14]:


pred


# In[16]:


frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()


# In[17]:


data['cluster']=pred


# In[18]:


data


# In[21]:


data[data['cluster']==2]


# In[11]:



                            
    


# In[ ]:




