#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# ## Step 1: collecting data

# In[96]:


#Reading data

titanic_data = pd.read_csv('titanic_train_clean.csv')
titanic_data.head(10)


# In[97]:


#Get the total number of passengers 
titanic_data.shape


# ## Step 2 Analyzing Data

# In[98]:


#survived vs not survived
sns.countplot(x = 'Survived', data = titanic_data)


# In[99]:


#how many males and females
sns.countplot(x = 'Survived', hue='Sex', data=titanic_data)


# In[100]:


#ticket class of the passengers
sns.countplot(x='Survived', hue='Pclass', data=titanic_data)


# In[101]:


#Age distribution
titanic_data['Age'].plot.hist()


# In[102]:


#Fare Distribution
titanic_data['Fare'].plot.hist()


# In[103]:


#get info on columns 
titanic_data.info()


# In[104]:


sns.countplot(x='SibSp', data=titanic_data)


# In[105]:


sns.countplot(x='Parch', data=titanic_data)


# ## Step 3 Data Wrangling

# In[106]:


#cleaning the data, removing all null values

#check for null values
titanic_data.isnull()


# In[107]:


#show all null values in the dataset
titanic_data.isnull().sum()


# In[108]:


#display in heatmap
sns.heatmap(titanic_data.isnull(), cmap="viridis")


# In[109]:


#if age had null
#plot boxplot for visualization

sns.boxplot(x='Pclass', y='Age', data=titanic_data)

#imputation
def impute_age(cols):
    Age =cols[0]
    Pclass = cols[6]
    if(pd.isnull(Age)):
        if(Pclass == 1):
            return 37
        elif(Pclass == 2):
            return 29
        else:
            return 24
    else:
        return Age
    
    
titanic_data['Age'] = titanic_data[['Age', 'Pclass'].apply(impute_age, axis = 1)]


# In[110]:


titanic_data.head(5)


# In[111]:


#Cabin column has too many null values, lets drop it
titanic_data.drop('Cabin', axis=1, inplace=True)


# In[112]:


sns.heatmap(titanic_data.isnull(), cmap="viridis")


# In[113]:


#drop all na values
titanic_data.dropna(inplace=True)


# In[114]:


#check if the dataset is clean
titanic_data.isnull().sum()


# In[115]:


#set gender to be binary
sex = pd.get_dummies(titanic_data['Sex'], drop_first = True)


# In[116]:


#apply get_dummies to embarked
embark = pd.get_dummies(titanic_data['Embarked'], drop_first=True)
embark.head(5)


# In[117]:


pcl = pd.get_dummies(titanic_data['Pclass'], drop_first=True)
pcl.head(5)


# In[118]:


title = pd.get_dummies(titanic_data['Title'])
title.head(5)


# In[119]:


#merge the data
titanic_data= pd.concat([titanic_data, sex, embark, pcl], axis= 1)
titanic_data.head(5)


# In[120]:



titanic_data.head(5)


# In[121]:


titanic_data.head(5)


# In[122]:


titanic_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)


# In[123]:


titanic_data.head(5)


# In[124]:


titanic_data.drop('Title', axis=1, inplace=True)


# In[125]:


titanic_data.head(5)


# In[126]:


titanic_data.drop(['Embarked', 'Pclass', 'Sex'], axis=1, inplace=True)


# In[127]:


titanic_data.head(5)


# ## Train and Test

# In[128]:


y = titanic_data['Survived']


# In[129]:


X = titanic_data.drop('Survived', axis=1)


# In[130]:


from sklearn.model_selection import train_test_split


# In[131]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[132]:


from sklearn.linear_model import LogisticRegression


# In[133]:


logmodel = LogisticRegression()


# In[135]:


logmodel.max_iter = 120000


# In[136]:


logmodel.fit(X_train, y_train)


# In[137]:


predictions = logmodel.predict(X_test)


# In[139]:


from sklearn.metrics import classification_report


# In[140]:


classification_report(y_test, predictions)


# In[141]:


from sklearn.metrics import confusion_matrix


# In[142]:


confusion_matrix(y_test, predictions)


# In[144]:


from sklearn.metrics import accuracy_score


# In[145]:


accuracy_score(y_test, predictions)


# In[146]:


accuracy_score(y_test, predictions) * 100


# In[ ]:




