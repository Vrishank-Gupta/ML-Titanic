#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


# In[16]:


train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


# In[17]:


train.head()


# In[18]:


test.head()


# In[19]:


train.info()


# In[20]:


test.info()


# In[29]:


g = sb.FacetGrid(train_dataset, col='Survived')
g.map(plt.hist, 'Age', bins=20)
g.map(plt.plot, "Age", "Survived", marker=".", color="red")


# In[33]:


plt.subplot2grid((3,4), (0,3))
train.Sex[train_dataset.Survived == 1].value_counts(normalize=True).plot(kind="pie")


# In[ ]:





# In[58]:


# train = train_dataset
train1 = train
train1["Hypothesis"] = 0
train1.loc[train.Sex == "female", "Hypothesis"] = 1

train1["Result"] = 0
train1.loc[train.Survived == train1["Hypothesis"], "Result"] = 1

print(train1["Result"].value_counts(normalize=True))


# In[82]:


def data_process(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())
    
    data = data.drop(['Ticket'], axis=1)
    data = data.drop(['Cabin'], axis=1)
    freq_port = train.Embarked.dropna().mode()[0]

    data['Embarked'] = data['Embarked'].fillna(freq_port)

    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    data = data.drop(['Name'], axis=1)
    
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
      
    data.loc[ data['Age'] <= 16, 'Age'] = int(0)
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age']
    
    return data
    
    
    


# In[83]:



train_dataset = data_process(train)
train_dataset.head()


# In[84]:


train_dataset.head()


# In[80]:


from sklearn import tree
from sklearn import linear_model, preprocessing

target = train_dataset["Survived"].values
features = train_dataset[["Pclass","Sex","Age","SibSp","Parch","Fare"]].values

decision_tree = tree.DecisionTreeClassifier(random_state = 1)
decision_tree_ = decision_tree.fit(features, target)

print(decision_tree_.score(features, target))


# In[ ]:





# In[ ]:





# In[85]:


def entropy(col):
    
    counts = np.unique(col,return_counts=True)
    N = float(col.shape[0])
    
    ent = 0.0
    
    for ix in counts[1]:
        p  = ix/N
        ent += (-1.0*p*np.log2(p))
    
    return ent


# In[87]:


def divide_data(x_data,fkey,fval):
    x_right = pd.DataFrame([],columns=x_data.columns)
    x_left = pd.DataFrame([],columns=x_data.columns)
    
    for ix in range(x_data.shape[0]):
        val = x_data[fkey].loc[ix]
        
        if val > fval:
            x_right = x_right.append(x_data.loc[ix])
        else:
            x_left = x_left.append(x_data.loc[ix])
            
    return x_left,x_right


# In[88]:


def information_gain(x_data,fkey,fval):
    
    left,right = divide_data(x_data,fkey,fval)
    
    #% of total samples are on left and right
    l = float(left.shape[0])/x_data.shape[0]
    r = float(right.shape[0])/x_data.shape[0]
    
    #All examples come to one side!
    if left.shape[0] == 0 or right.shape[0] ==0:
        return -1000000 #Min Information Gain
    
    i_gain = entropy(x_data.Survived) - (l*entropy(left.Survived)+r*entropy(right.Survived))
    return i_gain


# In[90]:


input_cols = ['Pclass',"Sex","Age","SibSp","Parch","Fare"]
output_cols = ["Survived"]

X = train_dataset[input_cols]
Y = train_dataset[output_cols]

print(X.shape,Y.shape)
print(type(X))


# In[92]:


for fx in X.columns:
    print(fx)
    print(information_gain(train_dataset,fx,train_dataset[fx].mean()))


# In[99]:


class DecisionTree:
    
    def __init__(self,depth=0,max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None
        
    def train(self,X_train):
        
        features = ['Pclass','Sex','Age','SibSp', 'Parch', 'Fare']
        info_gains = []
        
        for ix in features:
            i_gain = information_gain(X_train,ix,X_train[ix].mean())
            info_gains.append(i_gain)
            
        self.fkey = features[np.argmax(info_gains)]
        self.fval = X_train[self.fkey].mean()
        
        data_left,data_right = divide_data(X_train,self.fkey,self.fval)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)
         
        if data_left.shape[0]  == 0 or data_right.shape[0] ==0:
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survive"
            else:
                self.target = "Dead"
            return
        if(self.depth>=self.max_depth):
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survive"
            else:
                self.target = "Dead"
            return
        
        self.left = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(data_left)
        
        self.right = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(data_right)
        
        if X_train.Survived.mean() >= 0.5:
            self.target = "Survive"
        else:
            self.target = "Dead"
        return
    
    def predict(self,test):
        if test[self.fkey]>self.fval:
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
            if self.left is None:
                return self.target
            return self.left.predict(test)


# In[130]:


#Test Train Validation Split

split = int(0.7*train_dataset.shape[0])
train_data = train_dataset[:split]
test_data = train_dataset[split:]
test_data = test_data.reset_index(drop=True)


# In[131]:


tree = DecisionTree()


# In[139]:


tree.train(train_data)


# In[141]:


test.head()


# In[142]:


test = data_process(test)


# In[143]:


y_pred = []
for ix in range(test.shape[0]):
    y_pred.append(tree.predict(test.loc[ix]))


# In[144]:


y_pred


# In[145]:


from sklearn.preprocessing import LabelEncoder

y_actual = test_data[output_cols]
le = LabelEncoder()
y_pred = le.fit_transform(y_pred)
# y_pred = np.array(y_pred).reshape((-1,1))
print(y_pred.shape)

print(y_pred)
print(test_dataset.shape)


# In[146]:





# In[147]:


test_dataset_copy = pd.read_csv('./test.csv')
submission = pd.DataFrame({
        "PassengerId": test_dataset_copy["PassengerId"],
        "Survived": y_pred
})
# 
submission.to_csv('submission.csv', index=False)


# In[ ]:




