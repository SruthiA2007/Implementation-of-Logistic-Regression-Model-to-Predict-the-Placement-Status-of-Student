# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.

2.Print the present data and placement data and salary data.

3.Using logistic regression find the predicted values of accuracy confusion matrices.

4.Display the results. 
 

## Program Output::
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
```

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv("/content/Placement_Data.csv")

df

![image](https://github.com/user-attachments/assets/765d49d6-5977-438b-9efb-a68ee770e8a3)

df.info()

![Screenshot 2025-04-27 134036](https://github.com/user-attachments/assets/c782d847-ed84-4950-8df0-279af5b7fde1)

df=df.drop('salary',axis=1)

df.info()

![image](https://github.com/user-attachments/assets/f6f44dcb-80e6-4aa9-b4ac-152efeaeb2dd)

df['gender']=df['gender'].astype('category')

df['ssc_b']=df['ssc_b'].astype('category')

df['hsc_b']=df['hsc_b'].astype('category

df['degree_t']=df['degree_t'].astype("category

df['workex']=df['workex'].astype('category')

df['specialisation']=df['specialisation'].astype('category')

df['status']=df['status'].astype('category')

df['hsc_s']=df['hsc_s'].astype('category')

df.dtypes

![image](https://github.com/user-attachments/assets/b1b3efe4-1d94-416e-a80c-45055c727ede)

df.info()

![image](https://github.com/user-attachments/assets/4ec94725-24bb-4dd9-9a89-d52f23fdbe93)

df['gender']=df['gender'].cat.codes

df['ssc_b']=df['ssc_b'].cat.codes

df['hsc_b']=df['hsc_b'].cat.codes

df['degree_t']=df['degree_t'].cat.codes

df['workex']=df['workex'].cat.codes

df['specialisation']=df['specialisation'].cat.codes

df['status']=df['status'].cat.codes

df['hsc_s']=df['hsc_s'].cat.codes

df

![image](https://github.com/user-attachments/assets/5756d305-82f0-4b84-920b-efb0cb1f0eac)

df.info()

![image](https://github.com/user-attachments/assets/9646c55d-e334-414e-8b6e-762286de82dc)

x=df.iloc[:, :-1].values

y=df.iloc[:,-1].values

y

![image](https://github.com/user-attachments/assets/3061a3f9-8b1f-4ea0-8c0d-cf1680c188b0)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

print(x_train.shape)

print(x_test.shape)

![image](https://github.com/user-attachments/assets/23364548-8e02-400a-afa8-f62273b8a272)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(y_pred,y_test))

print(confusion_matrix(y_pred,y_test))

![image](https://github.com/user-attachments/assets/c0300c0e-bbe8-4b66-973b-c3e6367f866b)

 predict for new input

lr.predict([[0,87,0,95,0,2,8,0,0,1,5,6,5]])

lr.predict([[1,2,3,4,5,6,7,8,9,10,11,12,13]])

![image](https://github.com/user-attachments/assets/f9fb36ec-0767-4065-a37f-151808e46984)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
