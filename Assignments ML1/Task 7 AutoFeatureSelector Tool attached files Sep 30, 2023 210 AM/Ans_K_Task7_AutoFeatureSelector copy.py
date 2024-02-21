#!/usr/bin/env python
# coding: utf-8

# # Task 7: AutoFeatureSelector Tool
# ## This task is to test your understanding of various Feature Selection methods outlined in the lecture and the ability to apply this knowledge in a real-world dataset to select best features and also to build an automated feature selection tool as your toolkit
# 
# ### Use your knowledge of different feature selector methods to build an Automatic Feature Selection tool
# - Pearson Correlation
# - Chi-Square
# - RFE
# - Embedded
# - Tree (Random Forest)
# - Tree (Light GBM)

# ### Dataset: FIFA 19 Player Skills
# #### Attributes: FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats


# In[2]:


player_df = pd.read_csv("fifa19.csv")


# In[3]:


numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']


# In[4]:


player_df = player_df[numcols+catcols]


# In[5]:


traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
features = traindf.columns

traindf = traindf.dropna()


# In[6]:


traindf = pd.DataFrame(traindf,columns=features)


# In[7]:


y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']


# In[8]:


X.head()


# In[9]:


len(X.columns)


# ### Set some fixed set of features

# In[10]:


feature_name = list(X.columns)
# no of maximum features we need to select
num_feats=30


# ## Filter Feature Selection - Pearson Correlation

# ### Pearson Correlation function

# In[11]:


def cor_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    cor_list = []
    # Calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # Replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # Feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # Feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    # Your code ends here
    return cor_support, cor_feature


# In[12]:


cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')


# ### List the selected features from Pearson Correlation

# In[13]:


cor_feature


# ## Filter Feature Selection - Chi-Sqaure

# In[14]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


# ### Chi-Squared Selector function

# In[15]:


def chi_squared_selector(X, y, num_feats):
    # Your code ends here
    # Your code goes here (Multiple lines)
    # Convert data to be non-negative
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
     # Your code ends here
    return chi_support, chi_feature


# In[16]:


chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
print(str(len(chi_feature)), 'selected features')


# ### List the selected features from Chi-Square 

# In[17]:


chi_feature


# ## Wrapper Feature Selection - Recursive Feature Elimination

# In[18]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# ### RFE Selector function

# In[19]:


def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    # Initialize a model
    model = LogisticRegression(solver='liblinear')
    # Initialize RFE with step=10 to remove 10 features at each iteration
    rfe = RFE(estimator=model, n_features_to_select=num_feats, step=10, verbose=1)
    rfe_fit = rfe.fit(X, y)
    rfe_support = rfe_fit.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    # Your code ends here
    return rfe_support, rfe_feature


# In[20]:


rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
print(str(len(rfe_feature)), 'selected features')


# ### List the selected features from RFE

# In[21]:


rfe_feature


# ## Embedded Selection - Lasso: SelectFromModel

# In[22]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


# In[23]:


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    # Scale data to be between 0 and 1
    X_norm = MinMaxScaler().fit_transform(X)
    # Initialize a model with L1 penalty and a larger C for weaker regularization
    model = LogisticRegression(solver='liblinear', penalty="l1", C=0.5)
    model.fit(X_norm, y)
    # Get the model coefficients
    coef = model.coef_[0]
    # Sort the coefficients and select the top num_feats
    sorted_idx = np.argsort(np.abs(coef))[::-1][:num_feats]
    embedded_lr_support = np.zeros(X.shape[1], dtype=bool)
    embedded_lr_support[sorted_idx] = True
    embedded_lr_feature = X.columns[embedded_lr_support].tolist()
    # Your code ends here
    return embedded_lr_support, embedded_lr_feature


# In[24]:


embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
print(str(len(embedded_lr_feature)), 'selected features')


# In[25]:


embedded_lr_feature


# ## Tree based(Random Forest): SelectFromModel

# In[26]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[27]:


def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    # Initialize a RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100)
    # Initialize SelectFromModel
    sfm = SelectFromModel(rf_model, max_features=num_feats)
    sfm.fit(X, y)
    embedded_rf_support = sfm.get_support()
    embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature


# In[28]:


embedder_rf_support, embedder_rf_feature = embedded_rf_selector(X, y, num_feats)
print(str(len(embedder_rf_feature)), 'selected features')


# In[29]:


embedder_rf_feature


# ## Tree based(Light GBM): SelectFromModel

# In[48]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


# In[53]:


def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
     # Initialize an LGBMClassifier
# Initialize an LGBMClassifier
    lgbmc = LGBMClassifier(n_estimators=500,
                          learning_rate=0.05,
                          num_leaves=32,
                          colsample_bytree=0.2,
                          reg_alpha=3,
                          reg_lambda=1,
                          min_split_gain=0.01,
                          min_child_weight=40
    )

    embedded_lgbm_selector = SelectFromModel(lgbmc, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
  
    # Your code ends here
    return embedded_lgbm_support, embedded_lgbm_feature


# In[2]:


embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats=6)
print(str(len(embedded_lgbm_feature)), 'selected features')


# In[3]:


embedded_lgbm_feature


# ## Putting all of it together: AutoFeatureSelector Tool

# In[ ]:


pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


# ## Can you build a Python script that takes dataset and a list of different feature selection methods that you want to try and output the best (maximum votes) features from all methods?

# In[5]:


def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    
    # Your code ends here
    return X, y, num_feats


# In[6]:


def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    
    #### Your Code ends here
    return best_features


# In[7]:


best_features = autoFeatureSelector(dataset_path="data/fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
best_features


# ### Last, Can you turn this notebook into a python script, run it and submit the python (.py) file that takes dataset and list of methods as inputs and outputs the best features

# In[ ]:




