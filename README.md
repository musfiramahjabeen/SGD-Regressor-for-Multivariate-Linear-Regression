
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load necessary libraries like numpy, pandas, and StandardScaler for scaling features and target variables.

2.Implement a linear_regression function that uses gradient descent to minimize the cost function and compute optimal model parameters (theta).

3.Read the dataset from a CSV file, extract features (X) and target variable (y), and convert them to numeric types.

4.Scale both the feature matrix (X1) and target variable (y) using StandardScaler to improve gradient descent performance.

5.Call the linear_regression function with the scaled features and target to compute the model parameters (theta).

6.Scale new data using the same scaler, apply the model parameters (theta), and inverse scale the prediction to get the final result. 

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:   MUSFIRA MAHJBEEN M
RegisterNumber:  212223230130
*/
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("House vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,Y_pred,color="blue")
plt.title("House vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
HEAD VALUES
 
![image](https://github.com/user-attachments/assets/37c4b002-700b-4496-80c4-e61c92096d9f)

 
TAIL VALUES

![image](https://github.com/user-attachments/assets/ac036b86-d609-4e92-96bb-3fd9ba56a6ab)

HOURS VALUES
 
![image](https://github.com/user-attachments/assets/353de83d-c190-4b5e-ae93-2151a6640e4f)


SCORES VALUES
![image](https://github.com/user-attachments/assets/64384769-a131-4246-a47c-ac897b440553)


Y_PREDICTION

![image](https://github.com/user-attachments/assets/56229d37-81a2-40ff-8364-e18dacdaa090)


Y_TEST
 
![image](https://github.com/user-attachments/assets/6085a300-309e-4dca-88a6-b081925cc9a7)


RESULT OF MSE,MAE,RMSE

![image](https://github.com/user-attachments/assets/c56a8208-1263-4647-84ba-5725f16fb2e0)


TRAINING SET

![image](https://github.com/user-attachments/assets/b1f8d848-32d5-4390-8a25-4314017af519)

TEST SET

![image](https://github.com/user-attachments/assets/19baa062-3845-4036-b015-ec0abfa5e3bd)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
