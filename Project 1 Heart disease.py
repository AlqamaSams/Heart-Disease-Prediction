#!/usr/bin/env python
# coding: utf-8

# # first we will import the libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


heart_data=pd.read_csv(r'C:\Users\Zeeshan Sams\Downloads\test1.csv')


# In[3]:


heart_data.head()


# In[4]:


heart_data.tail()


# In[5]:


heart_data.describe() # this give the info about mean median sd and many more


# In[ ]:


#now we will see wheather if it have any null or zero values in the data


# In[6]:


heart_data.info() 


# also we can check 0 values by the code .isnull

# In[8]:


heart_data.isnull()


# # now after checking the null values and finding the data mean to max value we will analyze the data

# In[17]:


heart_data['target'].value_counts()


# In[19]:


heart_data['chol'].value_counts() #just checking to make sure


# now we will split the features and target because features ki help se hum target ko achieve kr sakte h

# # abb target ko features se ya data se alag kr denge or iss data ko alag alag variables se store krenge kyuki inn features ki help se hum target ko predict krenge 

# In[20]:


#now we will store them in different variables 


# In[23]:


x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']
print(x)
print(y)


# # abb hum data ko train or test m split kr denge 80-20 m pareto rule wala

# In[25]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)


# # abb hum training krenge data ki with the help of LogisticRegression

# In[26]:


#model training


# In[27]:


model=LogisticRegression()


# In[28]:


#training the logistic regression with training data


# In[29]:


model.fit(x_train, y_train)


# # 7 step model evaluation

# In[30]:


#accuracy score


# In[32]:


x_train_prediction= model.predict(x_train) #iski help se phir hum target data y_train pr predict krene isliye x_train dala h
training_data_accuracy= accuracy_score(x_train_prediction, y_train)
print('accuracy on training data: ',training_data_accuracy)


# In[34]:


x_test_prediction= model.predict(x_test) 
test_data_accuracy= accuracy_score(x_test_prediction, y_test)
print('accuracy on test data: ',test_data_accuracy)


# # 8 step predicting for any value we give in input

# In[40]:


input_data= (56,0,1,160,300,0,1,170,0,0.8,2,0,2)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction= model.predict(input_data_reshaped)
print(prediction)

if prediction==1:
    print('the patient is having heart disease')
else:
    print('the patient heart is healthy')


# In[ ]:




