# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Import the required libraries.

Load the car price dataset.

Separate the independent variables (X) and target variable (y).

Split the dataset into training and testing sets.

Create polynomial features of the required degree.

Build a pipeline for Ridge regression with polynomial features.

Build a pipeline for Lasso regression with polynomial features.

Build a pipeline for ElasticNet regression with polynomial features.

Train all three models using the training data.

Predict car prices using the test data.

Evaluate the models using MSE and R² score.

Compare the results and select the best model.
```
## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: 
RegisterNumber:  
*/
```
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by:roopika m
RegisterNumber: 25008774
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
data=pd.read_csv("encoded_car_data (1).csv")
data.head()
data = pd.get_dummies(data, drop_first=True)
x=data.drop('price',axis=1)
y=data['price']
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y.values.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet (alpha=1.0, l1_ratio=0.5)
}
result= {}
for name,model in models.items():
    pipeline = Pipeline([('poly',PolynomialFeatures(degree=2)),
    ('regressor',model)
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    result[name] = {'MSE': mse, 'MAE': mae, 'R² Score': r2}
print('Name:  Roopika m')
print('Reg. No: 25008774')
for model_name, metrics in result.items():
    print (f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f}, Mean Absolute Error: {metrics['MAE']:.2f}, R² Score: {metrics['R² Score']:.2f}")
results_df = pd.DataFrame(result).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis')
plt.title('R2 Score')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Show the plots
plt.tight_layout()
plt.show()
```

## Output:
<img width="1247" height="272" alt="image" src="https://github.com/user-attachments/assets/34564c91-6aa3-42a8-bd52-2768d6e5ecf1" />


<img width="1286" height="519" alt="image" src="https://github.com/user-attachments/assets/54bcaddc-db8e-47a2-b376-1328d3ed15ff" />


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
