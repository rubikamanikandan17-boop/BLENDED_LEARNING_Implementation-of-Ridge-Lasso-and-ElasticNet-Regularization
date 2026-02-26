# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("encoded_car_data (1).csv")
data.head()
diesel	gas	std	turbo	convertible	hardtop	hatchback	sedan	wagon	4wd	...	wheelbase	curbweight	enginesize	boreratio	horsepower	carlength	carwidth	citympg	highwaympg	price
0	0.0	1.0	1.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	...	88.6	2548.0	130.0	3.47	111.0	168.8	64.1	21.0	27.0	13495.0
1	0.0	1.0	1.0	0.0	1.0	0.0	0.0	0.0	0.0	0.0	...	88.6	2548.0	130.0	3.47	111.0	168.8	64.1	21.0	27.0	16500.0
2	0.0	1.0	1.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0	...	94.5	2823.0	152.0	2.68	154.0	171.2	65.5	19.0	26.0	16500.0
3	0.0	1.0	1.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	...	99.8	2337.0	109.0	3.19	102.0	176.6	66.2	24.0	30.0	13950.0
4	0.0	1.0	1.0	0.0	0.0	0.0	0.0	1.0	0.0	1.0	...	99.4	2824.0	136.0	3.19	115.0	176.6	66.4	18.0	22.0	17450.0
5 rows × 36 columns

data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
​
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}
​
results = {}
​
  # Train and evaluate each model
for name, model in models.items():
    # Create a pipeline with polynomial features and the model
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    predictions = pipeline.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Store results
    results[name] = {'MSE': mse, 'R² Score': r2}
​
# Print results
print('Name:Rubika m ')
print('Reg. No: 25008774')
for model_name, metrics in results.items():
    print(f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f}, R² Score: {metrics['R² Score']:.2f}")
​
# Visualization of the results
# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
​
# Set the figure size
plt.figure(figsize=(12, 5))
​
# Bar plot for MSE
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)  
 
  
Name:Rubika m 
Reg. No: 25008774
Ridge - Mean Squared Error: 0.26, R² Score: 0.79
Lasso - Mean Squared Error: 0.94, R² Score: 0.25
ElasticNet - Mean Squared Error: 0.63, R² Score: 0.49
(array([0, 1, 2]),
 [Text(0, 0, 'Ridge'), Text(1, 0, 'Lasso'), Text(2, 0, 'ElasticNet')])
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: 
RegisterNumber:  
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="494" height="513" alt="image" src="https://github.com/user-attachments/assets/7ff86e3b-8c09-4079-8d5c-681b9ada1280" />


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
