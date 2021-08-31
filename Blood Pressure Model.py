#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
df = pd.read_csv("FinalData.csv")
df.head()


# In[2]:


newdf=df.drop(['Stages','History','Patient','ControlledDiet','TakeMedication'], axis = 1)
newdf


# In[3]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[4]:


df2=(newdf.apply(le.fit_transform))

# In[5]:


result = pd.concat([df, df2], axis=1, join="inner")

# In[6]:


N=13
finalresult = result.iloc[: , N:]

# In[7]:


target = finalresult.Stages
inputs = finalresult.drop('Stages',axis='columns')
# In[8]:


x_train, x_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)
# In[9]:


len(x_train), len(y_train)


# In[10]:


len(x_test), len(y_test)


# In[11]:


model = MultinomialNB()
model.fit(x_train, y_train)

# In[12]:


print(model.score(x_test, y_test))


# In[13]:


y_pred = model.predict(x_test)
print(y_pred)


# In[14]:


acc=accuracy_score(y_test, y_pred)*100
print(acc)

# In[15]:


print(model.predict([[1,0,1,0,0,0,1,1,3]]))


# In[16]:


import pickle


# In[17]:


with open('model_pickle','wb') as file:
    pickle.dump(model,file)
    

# In[18]:


with open('model_pickle','rb') as file:
    mp= pickle.load(file)
    


# In[19]:


print(mp.score(x_test,y_test))

