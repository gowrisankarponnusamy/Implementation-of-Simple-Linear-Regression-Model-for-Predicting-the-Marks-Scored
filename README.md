# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1. Start the program.

STEP 2.Import the standard Libraries.

STEP 3.Set variables for assigning dataset values.

STEP 4.Import linear regression from sklearn.

STEP 5.Assign the points for representing in the graph.

STEP 6.Predict the regression for marks by using the representation of the graph.

STEP 7.End the program.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GOWRISANKAR P
RegisterNumber: 212222230041

```

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SEC/Downloads/student_scores.csv")
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
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)  
```

## Output:
### df.head() 
![image](https://github.com/user-attachments/assets/e785ff8b-7229-417e-87c9-d27d94faa451)

### df.tail()
![image](https://github.com/user-attachments/assets/c298474b-4a78-4ed4-92f2-c194e292edc3)

### Array value of x
![image](https://github.com/user-attachments/assets/7c51e600-21d3-442b-8216-1088d082d87a)

### Array valye of y
![image](https://github.com/user-attachments/assets/e7aa4960-132a-4b9a-9efa-3bb26870bc5c)

### Values of y prediction
![image](https://github.com/user-attachments/assets/ab5fc83c-eb27-4719-b36a-73bb048ed5e6)

### Array values of y test
![image](https://github.com/user-attachments/assets/3661ce0e-ba5a-4ebe-b34e-1fdc044a9d5f)

### Training set graph
![image](https://github.com/user-attachments/assets/196b0e66-48a3-4aff-b1ed-1edfab608ee3)

### Test  set graph
![image](https://github.com/user-attachments/assets/f0ef044c-a080-4d19-9270-69c400cd4484)

### Values of MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/1805e0bd-dac8-4046-a88a-191bdd4ebd64)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
