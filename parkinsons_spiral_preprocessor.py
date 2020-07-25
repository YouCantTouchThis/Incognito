#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "/Users/pahel/Desktop/7_25_Hacks/178338-401677-bundle-archive/spiral/training"
CATEGORIES = ["healthy" , "parkinson" ] 

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap = "gray")
        plt.show()


# In[2]:


print(img_array)


# In[3]:


IMG_SIZE = 15

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(img_array, cmap = "gray")
plt.show()


# In[4]:


print(img_array)


# In[5]:


training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()


# In[6]:


print(len(training_data))


# In[7]:


import random

random.shuffle(training_data)


# In[8]:


for sample in training_data:
    print(sample[1])


# In[11]:


X = []
y = []


# In[12]:


for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[13]:


import pickle 

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[14]:


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


# In[15]:


X[1]


# In[ ]:




