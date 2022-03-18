#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import Library:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# #### Data Collection and Processing

# In[2]:


# Loading the dataset:
df=pd.read_csv('census_income.csv')


# In[3]:


# showing 5 raws of data:
df.head()


# In[4]:


# Number of rows and columns
df.shape


# In[5]:


# information about full data
df.info()


# In[6]:


df.dtypes


# In[7]:


# checking the missing values:
df.isnull().sum()


# In[8]:


# checking the nunique values:
df.nunique()


# In[9]:


df.describe()


# #### Data Count values:

# In[10]:


df.columns


# In[11]:


df['Workclass'].value_counts()


# In[12]:


df['Occupation'].value_counts()


# In[13]:


df['Native_country'].value_counts()


# In[14]:


df['Marital_status'].value_counts()


# In[15]:


df['Sex'].value_counts()


# In[16]:


df['Race'].value_counts()


# In[17]:


df['Income'].value_counts()


# In[18]:


df['Education'].value_counts()


# #### Data Visualization:

# In[19]:


# display histogram:
df.hist(figsize=(12,12), layout=(3,3), sharex=False);


# In[20]:


## display countplot for Marital_status and Income:
plt.figure(figsize=(12,5))
sns.countplot(data = df, x = 'Marital_status', hue="Income", palette = 'nipy_spectral')
plt.xlabel("Marital_status", fontsize= 12)
plt.ylabel("No of people", fontsize= 12)
plt.ylim(0,5000) 
plt.show()


# In[21]:


# display countplot for Education and Sex:
sns.countplot(df['Education'], hue='Sex', data=df);


# In[22]:


# display countplot for Income and Relationship:
sns.countplot(df['Income'], palette='coolwarm', hue='Relationship', data=df);


# In[24]:


# display Boxplot:
df.plot(kind='box', figsize=(12,12), layout=(3,3), sharex=False, subplots=True);


# #### Data Correlation:

# In[25]:


df.corr()


# In[26]:


# display heatmap:
plt.subplots(figsize=(10,5))
sns.heatmap(df.corr(), annot=True);


# In[27]:


# finding ? value:
df.isin([' ?']).sum()


# In[28]:


df.replace(' ?',0,inplace=True)
Data = df.replace(0,np.nan)


# In[29]:


df.head()


# In[30]:


# Feature Scaling data:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Income'] = le.fit_transform(df['Income'])


# In[31]:


for col in df.columns:
    if df[col].dtypes == 'object':
        le = LabelEncoder()        
        df[col] = le.fit_transform(df[col].astype(str))


# In[32]:


data = df.replace(np.nan,0)


# In[33]:


data.head()


# In[34]:


# Seperating  the data and label:
X= df.drop(['Income'], axis=1)
y = df['Income']


# In[51]:


print(X)
print(y)


# #### Split datasets into train and test

# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)


# In[36]:


X_train.shape


# In[37]:


X_test.shape


# #### Creating Model:

# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()

model = lr.fit(X_train, y_train)
prediction = model.predict(X_test)

print("Acc on training data: {:,.3f}".format(lr.score(X_train, y_train)))
print("Acc on test data: {:,.3f}".format(lr.score(X_test, y_test)))


# In[39]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

model1 = rfc.fit(X_train, y_train)
prediction1 = model1.predict(X_test)

print("Acc on training data: {:,.3f}".format(rfc.score(X_train, y_train)))
print("Acc on test data: {:,.3f}".format(rfc.score(X_test, y_test)))


# In[40]:


from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier( max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
# acc_clf = round(clf.score(X_train, y_train) * 100, 2)
acc_clf = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
print("Acc on:",metrics.accuracy_score(y_test, y_pred))


# In[41]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test) 
# acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn = round(metrics.accuracy_score(y_test, y_pred) * 100, 2)
print("Acc on:",metrics.accuracy_score(y_test, y_pred))


# #### Model Evaluation:

# In[42]:


# Carting Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(y_test, prediction1))


# In[43]:


print(classification_report(y_test, prediction1))


# In[44]:


print('Precision =' , 6925/(6925+869))
print('Recall =', 6925/(6925+487))

#for other class : 1 (>50K)

print('Precision = ', 1487/(1487+487))
print('Recall= ', 1487/(1487+869))


# In[45]:


# here doing Cross validated:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
scores


# In[54]:


# Creating the hyperparameter grid:
from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
c_space = np.logspace(-5, 5, 8)
param_grid = {'C': c_space}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv = 5)
logreg_cv.fit(X,y)
print("Tuned DecisionTreeClassifier: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


# In[47]:


# here ploting roc_curve :
from sklearn import datasets, metrics, model_selection, svm
X, y = datasets.make_classification(random_state=0)
clf = svm.SVC(random_state=0)
clf.fit(X_train, y_train)
metrics.plot_roc_curve(clf, X_test, y_test) 
plt.show()


# In[48]:


# save the model:
import pickle
file = 'Census Income Project'
#save file
save = pickle.dump(model,open(file,'wb'))


# In[ ]:




