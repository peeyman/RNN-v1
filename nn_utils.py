
# coding: utf-8

# In[4]:


import numpy as np


# In[5]:


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# In[6]:


def sigmoid(x):
    return 1/(1+np.exp(-x))

