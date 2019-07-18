
# coding: utf-8

# In[159]:


import pandas as pd
import numpy as np
import sklearn
from openpyxl import load_workbook
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy import misc
import graphviz
import pydotplus
from sklearn.datasets.samples_generator import make_blobs
import io
import matplotlib.patches as mpatches
import random
import math
import xlrd
from itertools import product
from sklearn.model_selection import KFold 
from sklearn.cross_validation import cross_val_score
from xlrd import open_workbook
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt 
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[160]:


from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn import datasets


# In[161]:


data=pd.read_csv('./winequality-white.csv')


# In[39]:


data


# In[169]:


#iteration 1
for column in data:
    if str(column) != 'quality':
        print('Parameters, R^2 and AIC values for model with '+ column +':\n')
        lmwhite = smf.ols(formula='quality'+ '~' + column , data=data).fit()
        print('Parameters:\n')
        print(lmwhite.params)
        print('R Squared values:  ')
        print(lmwhite.rsquared)
        print('AIC Value:   ')
        print(lmwhite.aic)
        print('\n')
        
        


# In[171]:


#iteration 2
import pandas as pd
import statsmodels.formula.api as smf

data = pd.read_csv('winequality-white.csv')
data1 = data.iloc[:,data.columns!='quality']


for column in data1:
    if str(column) != 'alcohol':
        print('Parameters, R^2 and AIC values for model with '+ 'alcohol' + ' and '+column +':\n')
        lmwhite = smf.ols(formula='quality'+ '~' + 'alcohol' + '+'+ column , data=data).fit()
        print('Parameters:\n' + str(lmwhite.params))
        print('R Square : ' + str(lmwhite.rsquared))
        print('AIC: ' +  str(lmwhite.aic))
        print("\n")


# In[172]:


#iteration 3
import pandas as pd
import statsmodels.formula.api as smf

data = pd.read_csv('winequality-white.csv')
data1 = data.iloc[:,data.columns!='quality']


for column in data1:
    if str(column) != 'alcohol' and str(column)!= 'volatile_acidity':
        print('Parameters, R^2 and AIC values for model with '+ 'alcohol, volatile_acidity' + ' and '+column +':\n')
        lmwhite = smf.ols(formula='quality'+ '~' + 'alcohol' + '+' + 'volatile_acidity' + '+'+ column , data=data).fit()
        print('Parameters:\n' + str(lmwhite.params))
        print('R Square : ' + str(lmwhite.rsquared))
        print('AIC: ' +  str(lmwhite.aic))
        print("\n")


# In[174]:


#iteration 4
for column in datax:
    if str(column) != 'alcohol' and str(column)!= 'volatile_acidity' and str(column)!='residual_sugar':
        print('Parameters, R^2 and AIC values for model with '+ 'alcohol, volatile_acidity,residual_sugar' + ' and '+column +':\n')
        lmwhite = smf.ols(formula='quality'+ '~' + 'alcohol' + '+' + 'volatile_acidity' + '+' +'residual_sugar'+ '+' + column , data=data).fit()
        print('Parameters:\n' + str(lmwhite.params))
        print('R Square : ' + str(lmwhite.rsquared))
        print('AIC: ' +  str(lmwhite.aic))
        print("\n")


# In[175]:


#iteration 5 
for column in datax:
    if str(column) != 'alcohol' and str(column)!= 'volatile_acidity' and str(column)!='residual_sugar' and str(column)!='free_sulfur_dioxide':
        print('Parameters, R^2 and AIC values for model with '+ 'alcohol, volatile_acidity,residual_sugar,free_sulfur_dioxide' + ' and '+column +':\n')
        lmwhite = smf.ols(formula='quality'+ '~' + 'alcohol' + '+' + 'volatile_acidity' + '+' +'residual_sugar'+ '+' + 'free_sulfur_dioxide'+ '+' + column , data=data).fit()
        print('Parameters:\n' + str(lmwhite.params))
        print('R Square : ' + str(lmwhite.rsquared))
        print('AIC: ' +  str(lmwhite.aic))
        print("\n")


# In[177]:


#iteration 6
for column in datax:
    if str(column) != 'alcohol' and str(column)!= 'volatile_acidity' and str(column)!='residual_sugar' and str(column)!='free_sulfur_dioxide' and str(column)!='density':
        print('Parameters, R^2 and AIC values for model with '+ 'alcohol, volatile_acidity,residual_sugar,free_sulfur_dioxide,density' + ' and '+column +':\n')
        lmwhite = smf.ols(formula='quality'+ '~' + 'alcohol' + '+' + 'volatile_acidity' + '+' +'residual_sugar'+ '+' + 'free_sulfur_dioxide'+ '+' + 'density'+ '+' + column , data=data).fit()
        print('Parameters:\n' + str(lmwhite.params))
        print('R Square : ' + str(lmwhite.rsquared))
        print('AIC: ' +  str(lmwhite.aic))
        print("\n")


# In[179]:


#iteration 7
for column in datax:
    if str(column) != 'alcohol' and str(column)!= 'volatile_acidity' and str(column)!='residual_sugar' and str(column)!='free_sulfur_dioxide' and str(column)!='density' and str(column)!='pH':
        print('Parameters, R^2 and AIC values for model with '+ 'alcohol, volatile_acidity,residual_sugar,free_sulfur_dioxide,density,pH' + ' and '+column +':\n')
        lmwhite = smf.ols(formula='quality'+ '~' + 'alcohol' + '+' + 'volatile_acidity' + '+' +'residual_sugar'+ '+' + 'free_sulfur_dioxide'+ '+' + 'density'+ '+' + 'pH' + '+' + column , data=data).fit()
        print('Parameters:\n' + str(lmwhite.params))
        print('R Square : ' + str(lmwhite.rsquared))
        print('AIC: ' +  str(lmwhite.aic))
        print("\n")


# In[184]:


#iteration 8
for column in datax:
    if str(column) != 'alcohol' and str(column)!= 'volatile_acidity' and str(column)!='residual_sugar' and str(column)!='free_sulfur_dioxide' and str(column)!='density' and str(column)!='pH' and str(column)!='sulphates':
        print('Parameters, R^2 and AIC values for model with '+ 'alcohol, volatile_acidity,residual_sugar,free_sulfur_dioxide,density,pH,sulphates' + ' and '+column +':\n')
        lmwhite = smf.ols(formula='quality'+ '~' + 'alcohol' + '+' + 'volatile_acidity' + '+' +'residual_sugar'+ '+' + 'free_sulfur_dioxide'+ '+' + 'density'+ '+' + 'pH' + '+' + 'sulphates' + '+' + column , data=data).fit()
        print('Parameters:\n' + str(lmwhite.params))
        print('R Square : ' + str(lmwhite.rsquared))
        print('AIC: ' +  str(lmwhite.aic))
        print("\n")


# In[186]:


#iteration 9
for column in datax:
    if str(column) != 'alcohol' and str(column)!= 'volatile_acidity' and str(column)!='residual_sugar' and str(column)!='free_sulfur_dioxide' and str(column)!='density' and str(column)!='pH' and str(column)!='sulphates' and str(column)!='fixed_acidity' :
        print('Parameters, R^2 and AIC values for model with '+ 'alcohol, volatile_acidity,residual_sugar,free_sulfur_dioxide,density,pH,sulphates,fixed_acidity' + ' and '+column +':\n')
        lmwhite = smf.ols(formula='quality'+ '~' + 'alcohol' + '+' + 'volatile_acidity' + '+' +'residual_sugar'+ '+' + 'free_sulfur_dioxide'+ '+' + 'density'+ '+' + 'pH' + '+' + 'sulphates' + '+' + 'fixed_acidity' + '+' + column , data=data).fit()
        print('Parameters:\n' + str(lmwhite.params))
        print('R Square : ' + str(lmwhite.rsquared))
        print('AIC: ' +  str(lmwhite.aic))
        print("\n")


# In[221]:


lmwhite_best = smf.ols(formula='quality'+ '~' + 'alcohol' + '+' + 'volatile_acidity' + 
                  '+' +'residual_sugar'+ '+' + 'free_sulfur_dioxide'+ '+'
                  + 'density'+ '+' + 'pH' + '+' + 'sulphates' + 
                  '+' + 'fixed_acidity' , data=data).fit()
act_tar = data.iloc[:,-1]
#print(actual)
pred_tar = lmwhite_best.predict(data.iloc[:,:-1])
#print(pred)
diff_tar = act_tar - pred_tar

abs_diff = np.abs(diff_tar)

sorted_abs_diff = sorted(abs_diff)
worst_five_wines = sort_abs_diff[-5:]

five_indices = []
for x in worst_five_wines:
    for i,y in enumerate(abs_diff):
        if x==y:
            five_indices.append(i)
            break
print('Values of the absolute difference for the five wines that have the largest magnitudes of difference between the predicted and the actual wine-quality values:\n')
print(five_wines)
print('\n')
print('Indices of them in the given data:\n')
print(five_indices)
print('\n')
five_samples=[]
for x in five_indices:
    five_samples.append(x+2)
print('Locations of them as per the excel file:\n')
print(five_samples)

