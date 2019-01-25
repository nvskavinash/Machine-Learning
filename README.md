# Linear-Regression
Regression Analysis on a wine dataset
A wine data set is taken from URL Machine Learning Repository database: https://archive.ics.uci.edu/ml/datasets/wine+quality
Using Ols (Ordinary Least Square method), a linear regression model has been constructed.
The Linear Regression model has been constructed in a way by adding a single feature at a time and the feature with highest Rsquare and lowest AIC value is taken and is combined with other features.
This process is repeated until the stoping criterion is met, where the stoping criterion is occured when we observe an increment in the AIC value. 
So the combination of features that we got in the previous iteration is chosen as the best combination.
Once the linear regression model has been created the top 5 ouliers are displayed where the outliers are calculated using the Sum of Squated Error
