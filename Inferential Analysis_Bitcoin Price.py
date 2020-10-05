#!/usr/bin/env python
# coding: utf-8

# In[64]:


#####Inferential Analysis####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import cluster 
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage
import pylab as pl
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


# In[65]:


df = pd.read_csv("/Users/newuser/Downloads/data.csv")


# In[38]:


df.head()


# In[82]:


df.shape


# In[83]:


sns.pairplot(df)


# In[48]:


# x = feature, y = terget
X = df['BTC_network_hashrate'].values
#X = df['Nasdaq_composite_index'].values
#X = df['Estimated_TX_Volume_USD_BTC'].values
y = df['BTC_Price'].values


# In[49]:


# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))


# In[50]:


# Reshape X and y
y = y.reshape(-1, 1)
X = X.reshape(-1, 1)


# In[51]:


# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))


# In[31]:


pl.figure(figsize=(12,8))

corr = df.corr() 
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation between Factors')
plt.show()


# In[ ]:


#### Machine Learning Models ####


# In[84]:


#### Linear Regression #####

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=32)


# In[57]:


# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#mse = mean_squared_error(y_test, y_pred)
print("Root Mean Squared Error of Linear Regression: {}".format(rmse))
#print("Mean Squared Error of Linear Regression: {}".format(mse))


# In[58]:


#### Mean Absolute Error 
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error is {}'.format(mae))


# In[59]:



# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

# Fit the model to the data
reg.fit(X, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
# print(X_test)
# print(y_test)
# plt.plot(X_test,forecast_set,color='blue', linewidth=1)
plt.scatter(X, y, color='darkorange', label='data')
plt.title("Linear Regression")
plt.xlabel("DJI")
plt.ylabel("BTC Price")
# plt.plot(X_test, y_test, color='darkorange', linewidth=1)
# plt.plot(X_test, forecast_prediction, color='blue', linew/idth=2)
plt.show()


# In[14]:


#####Decison Tree#####

# Instantiate dt
dt = DecisionTreeRegressor()
    
# Fit dt to the training set
dt.fit(X_train, y_train)

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(dt.score(X_test, y_test)))
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error of DecisionTree : {}".format(rmse_dt))


# In[15]:


dt.get_params()


# In[16]:


###manually set the grid of hyperparameters###
# Define the dictionary 'params_rf'
params_dt = {
                'max_depth': [2, 4, 6, 8,10],
                'max_features':['log2', 'auto', 'sqrt'],
                'min_samples_leaf':[ 2, 4, 6, 8,10]
                }


# In[17]:


# Import GridSearchCV
from sklearn.model_selection import  GridSearchCV

# Instantiate grid_rf
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='neg_mean_squared_error',
                       cv=4,
                       verbose=1,
                       n_jobs=-1)


# In[18]:


# Fit 'grid_dt' to the training set
grid_dt.fit(X_train, y_train)


# In[19]:


# Extract best hyperparameters from 'grid_dt'
best_hyperparams = grid_dt.best_params_
print('Best hyerparameters:\n'
, best_hyperparams)


# In[20]:


###valuate the test set RMSE of grid_dt's optimal model.

from sklearn.metrics import mean_squared_error as MSE

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict test set labels
y_pred_dt= best_model.predict(X_test)

# Compute rmse_test
rmse_test_dt = MSE(y_test, y_pred_dt)**(1/2)

# Print rmse_test
print('Test RMSE of best model of Decsion Tree: {:.3f}'.format(rmse_test_dt)) 


# In[21]:


rmse_dt/rmse_test_dt-1


# In[22]:


#### Mean Absolute Error 
mae = mean_absolute_error(y_test, y_pred)
print('Untunned Mean Absolute Error is {}'.format(mae))


# In[23]:


#### Mean Absolute Error 
mae_t = mean_absolute_error(y_test, y_pred_dt)
print('Mean Absolute Error is {}'.format(mae_t))


# In[24]:


mae/mae_t-1


# In[25]:


# Create the prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

# Fit the model to the data
dt.fit(X, y)

# Compute predictions over the prediction space: y_pred
y_pred = dt.predict(prediction_space)

# Print R^2 
print(dt.score(X, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
# print(X_test)
# print(y_test)
# plt.plot(X_test,forecast_set,color='blue', linewidth=1)
plt.scatter(X, y, color='darkorange', label='data')
plt.title("Decsion Tree Regression")
plt.xlabel("DJI")
plt.ylabel("BTC Price")
# plt.plot(X_test, y_test, color='darkorange', linewidth=1)
# plt.plot(X_test, forecast_prediction, color='blue', linew/idth=2)
plt.show()


# In[85]:


####RandomForest#####

rf = RandomForestRegressor(
    max_depth=8, max_features='sqrt', min_samples_leaf=4, n_estimators=400,
    random_state = 2)
    
 ###{'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'n_estimators': 400}
###n_estimators=200, 

#rf = RandomForestRegressor(random_state=2)
            
# Fit rf to the training set    
rf.fit(X_train, y_train) 

# Predict the test set labels
y_pred = rf.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(rf.score(X_test, y_test)))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error of Random Forest_Best: {}".format(rmse_rf))


# In[86]:


rf.get_params()


# In[87]:


###manually set the grid of hyperparameters###

# Define the dictionary 'params_rf'
params_rf = {
                'max_depth': [2, 4, 6,8,10],
                'n_estimators':[100, 200, 400],
                'max_features':['log2', 'auto', 'sqrt'],
                'min_samples_leaf':[ 2, 4,6,8, 10]
                }


# In[88]:


# Import GridSearchCV
from sklearn.model_selection import  GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=4,
                       verbose=1,
                       n_jobs=-1)


# In[89]:


grid_rf


# In[1799]:


# Fit 'grid_rf' to the training set
grid_rf.fit(X_train, y_train)


# In[1800]:


# Extract best hyperparameters from 'grid_rf'
best_hyperparams = grid_rf.best_params_
print('Best hyerparameters:\n'
, best_hyperparams)


# In[1801]:


###valuate the test set RMSE of grid_rf's optimal model.

from sklearn.metrics import mean_squared_error as MSE

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred_rf= best_model.predict(X_test)

# Compute rmse_test
rmse_test_rf = MSE(y_test, y_pred_rf)**(1/2)

# Print rmse_test
print('Test RMSE of best model of RandomForest: {:.3f}'.format(rmse_test_rf)) 


# In[1802]:


rmse_rf/rmse_test_rf-1


# In[1803]:


#### Mean Absolute Error 
mae = mean_absolute_error(y_test, y_pred)
print('Untunned Mean Absolute Error is {}'.format(mae))


# In[1804]:


#### Mean Absolute Error 
mae_t = mean_absolute_error(y_test, y_pred_rf)
print('Mean Absolute Error is {}'.format(mae_t))


# In[1805]:


mae_t/mae -1


# In[1806]:


# Create the prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

# Fit the model to the data
rf.fit(X, y)

# Compute predictions over the prediction space: y_pred
y_pred = rf.predict(prediction_space)

# Print R^2 
print(rf.score(X, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
# print(X_test)
# print(y_test)
# plt.plot(X_test,forecast_set,color='blue', linewidth=1)
plt.scatter(X, y, color='darkorange', label='data')
plt.title("Random Forest Regression")
plt.xlabel("DJI")
plt.ylabel("BTC Price")
# plt.plot(X_test, y_test, color='darkorange', linewidth=1)
# plt.plot(X_test, forecast_prediction, color='blue', linew/idth=2)
plt.show()


# In[1706]:


####GradientBoost####

# Instantiate gb
gb = GradientBoostingRegressor()

#max_depth=2, max_features ='log2',
                               #n_estimators=100, min_samples_leaf=10,
                               #random_state = 2
            
# Fit rf to the training set    
gb.fit(X_train, y_train) 

# Predict the test set labels
y_pred = gb.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(gb.score(X_test, y_test)))
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error of GradientBoost: {}".format(rmse_gb))


# In[1708]:


gb.get_params()


# In[1709]:


###manually set the grid of hyperparameters###
# Define the dictionary 'params_gb'
params_gb = {
                'max_depth': [2, 4, 6, 8, 10],
                'n_estimators':[100, 200, 400],
                'max_features':['log2', 'auto', 'sqrt'],
                'min_samples_leaf':[ 2, 4, 6, 8, 10],
                }


# In[1710]:


# Instantiate grid_gb
grid_gb = GridSearchCV(estimator=gb,
                       param_grid=params_gb,
                       scoring='neg_mean_squared_error',
                       cv=4,
                       verbose=1,
                       n_jobs=-1)


# In[1711]:


# Fit 'grid_gb' to the training set
grid_gb.fit(X_train, y_train)


# In[1712]:


# Extract best hyperparameters from 'grid_gb'
best_hyperparams = grid_gb.best_params_
print('Best hyerparameters:\n'
, best_hyperparams)


# In[1713]:


# Extract the best estimator
best_model = grid_gb.best_estimator_

# Predict test set labels
y_pred_gb= best_model.predict(X_test)

# Compute rmse_test
rmse_test_gb = MSE(y_test, y_pred_gb)**(1/2)

# Print rmse_test
print('Test RMSE of best model of GBoost: {:.3f}'.format(rmse_test_gb))


# In[1719]:


rmse_gb/rmse_test_gb-1


# In[1720]:


#### Mean Absolute Error 
mae = mean_absolute_error(y_test, y_pred)
print('Untunned Mean Absolute Error is {}'.format(mae))


# In[1721]:


#### Mean Absolute Error 
mae_t = mean_absolute_error(y_test, y_pred_gb)
print('Mean Absolute Error is {}'.format(mae_t))


# In[1716]:


mae_t/mae -1


# In[1722]:


# Create the prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

# Fit the model to the data
gb.fit(X, y)

# Compute predictions over the prediction space: y_pred
y_pred = gb.predict(prediction_space)

# Print R^2 
print(gb.score(X, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
# print(X_test)
# print(y_test)
# plt.plot(X_test,forecast_set,color='blue', linewidth=1)
plt.scatter(X, y, color='darkorange', label='data')
plt.title("GradientBoost Regression")
plt.xlabel("DJI")
plt.ylabel("BTC Price")
# plt.plot(X_test, y_test, color='darkorange', linewidth=1)
# plt.plot(X_test, forecast_prediction, color='blue', linew/idth=2)
plt.show()


# In[ ]:


#### Bitcoin Price Prediction on the next 320 days####


# In[68]:


df_d.shape


# In[91]:


df_d = df[["BTC_Price"]]
df_d.head(5)


# In[433]:


##Create a variable to predict 'X' day out into the future
future_days = 500
## create a new column (target) shifted 'x' units/days up
df_d['Prediction'] = df_d[["BTC_Price"]].shift(-future_days)
df_d.head(10)


# In[434]:


#create the feature data set (x) and convert it to a numpy
X = np.array(df_d.drop(['Prediction'],1))[:-future_days]


# In[435]:


#create the target data set (y) and convert it to a numpy array adn get all of the target 
y = np.array(df_d['Prediction'])[:-future_days]


# In[436]:


###Split the data into 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=32)


# In[437]:


#### Set Randam Forest as chosen Machine Learning Model on prediction ####

### create the RFM
rf = grid_rf.fit(X_train, y_train) 

#rf = rf.fit(X_train, y_train) 
### Create LR
#lr = reg_all.fit(X_train, y_train)


# In[438]:


best_hyperparams = grid_rf.best_params_
print('Best hyerparameters:\n'
      
, best_hyperparams)


# In[439]:


best_model = grid_rf.best_estimator_


# In[440]:


best_model


# In[441]:


x_future = df_d.drop(['Prediction'],1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future


# In[442]:


#### RMSE of Machine Learning model  ####

rf_prediction = best_model.predict(x_future)


# In[443]:


rmse_test_rfp = MSE(x_future, rf_prediction)**0.5


# In[444]:


rmse_test_rfp


# In[445]:


#show the lr prediction
lr_prediction = lr.predict(x_future)


# In[448]:


#### Machine Learning model Visualization ####

predictions = rf_prediction
valid = df_d[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Machine Learning model - 320 days')
plt.xlabel('Days')
plt.ylabel('Bitcoin price')
plt.plot(df_d['BTC_Price'])
plt.plot(valid[['BTC_Price','Predictions']])
plt.legend(['Org', 'Val', 'Pred'])
plt.show
            


# In[449]:


#### Linear Regression model Visualization ####

predictions = lr_prediction
valid = df_d[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Linear Regression - 320 days')
plt.xlabel('Days')
plt.ylabel('Bitcoin price')
plt.plot(df_d['BTC_Price'])
plt.plot(valid[['BTC_Price','Predictions']])
plt.legend(['Org', 'Val', 'Pred'])
plt.show


# In[450]:


#### RMSE of  Linear Regression model  ####

rmse_test_LR = MSE(x_future, lr_prediction)**(1/2)


# In[451]:


rmse_test_LR

