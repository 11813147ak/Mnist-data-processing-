#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


read=pd.read_csv("C:/Users/HP/Desktop/mnist_train.csv")


# In[3]:


read.head(n=5)


# In[7]:


read.shape


# In[ ]:





# In[12]:


print(type(read))


# In[13]:


data=read.values


# In[14]:


data


# In[15]:


print(type(data))


# In[16]:


np.random.shuffle(data)


# In[17]:


X=data[:,1:]
Y=data[:,0]


# In[18]:


X[0].shape


# In[24]:


def drawimg(X,Y,i):
    plt.title("Label"+str(Y[i]))
    plt.imshow(X[i].reshape(28,28),cmap='gray')
    plt.show()
    
for i in range(5):
    drawimg(X,Y,i)


# In[25]:


XT,Xt,YT,Yt=train_test_split(X,Y,test_size=0.2)


# In[26]:


print(XT.shape,YT.shape)
print(Xt.shape,Yt.shape)


# In[32]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(X[i].reshape(28,28),cmap='gray')
    plt.axis("off")
    plt.title(Y[i])
    plt.show()


# In[ ]:




