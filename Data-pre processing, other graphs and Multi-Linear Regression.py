#!/usr/bin/env python
# coding: utf-8

# In[318]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm


# In[319]:


df = pd.read_csv('C:/Users/alaks/OneDrive/Documents/MLfiles/insurance.csv')

print(df)


# In[320]:



plt.scatter(df['bmi'],df['charges'])
plt.title('Charge by BMI')
plt.xlabel('BMI')
plt.ylabel('Charge')
plt.legend(labels=["Non-Smoker","Smoker"])


# In[321]:


f = plt.figure(figsize=(14,6))
ax = f.add_subplot(122)
sns.scatterplot(x = 'bmi', y = 'charges', data = df, hue='smoker' )


# In[322]:


df.query('bmi > 50 ')['age']


# In[323]:


df.drop(df.query('bmi > 50 ').index ,axis= 0 ,inplace=True)


# In[324]:


df.query('bmi > 50 ')['age']


# In[ ]:





# In[325]:


df.info()


# In[326]:


plt.scatter(df['charges'],df['smoker'])


# In[327]:


plt.figure(figsize=(16,8))
sns.heatmap(df.corr() ,annot=True)


# In[328]:



fig= plt.figure(figsize=(12,8))


sns.distplot(df[(df.smoker == 1)]["charges"])


sns.distplot(df[(df.smoker == 0)]['charges'])

plt.legend(labels=["Smoker","Non-Smoker"])


# In[329]:


df[df.duplicated()]


# In[330]:


region_cost= df.groupby('region')['charges'].sum() * 1e-6
fig = plt.figure(figsize=(16,8))
sns.barplot(region_cost.index , region_cost.values)
plt.title('Region Costs In Million')
plt.ylabel('(M)')
plt.show()


# In[331]:


region = pd.get_dummies(df['region'],drop_first = False)
df = pd.concat([df,region],axis = 1)
df.info()


# In[332]:


smoke = pd.get_dummies(df['smoker'],drop_first = True)
df = pd.concat([df,smoke],axis = 1)


# In[333]:


df = df.rename(columns={'yes':'Smoker'})


# In[334]:


sex = pd.get_dummies(df['sex'],drop_first = True)
df = pd.concat([df,sex],axis = 1)
df.info()


# In[ ]:





# In[335]:


df.head()


# In[336]:


df = df.drop(['sex','smoker','region'], axis = 1)
df.head()


# In[ ]:





# In[341]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = df.drop(['charges'],axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
l_g = LinearRegression()
l_g.fit(X_train, y_train)

print(l_g.intercept_)


# In[342]:


coeffecients = pd.DataFrame(l_g.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[343]:


predictions = l_g.predict(X_test)


# In[344]:


from sklearn import metrics
from sklearn.metrics import r2_score


plt.scatter(predictions,y_test)
plt.ylabel('Charges')
plt.xlabel('Predictions')


print('R2 Score for Linear Regression on test data: {}'.format( np.round(r2_score(y_test, predictions), 2)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




