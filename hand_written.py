#!/usr/bin/env python
# coding: utf-8

# In[13]:


#importing libs 
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
from sklearn.model_selection import train_test_split


# In[55]:


#Getting data
digits = datasets.load_digits()

#Reshaping the images in an array
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# In[62]:


#Splitting the data set in train and test
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False)


# In[30]:


#Fitting and Prediction
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 0.7, random_state = 42, max_iter = 20000)
lr.fit(X_train, y_train)
lr_prediction = lr.predict(X_test)


# In[31]:


#Calculatting accuracy
from sklearn import metrics
print("{0:.4f}".format(metrics.accuracy_score(y_test, lr_prediction)))


# In[53]:


#Print a set of numbers
import matplotlib.pyplot as plt
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test[100:], lr_prediction[100:]):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title(f"Prediction: {prediction}")


# In[65]:


from sklearn import svm
#Using  support vector classifier
svc = svm.SVC(gamma=0.001)
svc.fit(X_train, y_train)
svc_predicted = svc.predict(X_test)


# In[66]:


#Calculatting accuracy
from sklearn import metrics
print("{0:.4f}".format(metrics.accuracy_score(y_test, svc_predicted)))


# In[ ]:




