import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Breast_cancer_data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

# from sklearn.metrics import confusion_matrix, accuracy_score
# cm=confusion_matrix(y_test,y_pred)
# print(cm)
# accuracy_score(y_test,y_pred)

while True:
    print("1. ENTER DETAILS.")
    print("2.EXIT")
    x=int(input("Enter your option:"))
    if(x==1):
        r=float(input("Enter the mean radius:"))
        t=float(input("Enter the mean texture"))
        p=float(input("Enter the mean perimeter:"))
        a=float(input("Enter the mean area:"))
        s=float(input("Enter the smoothness:"))

        ans=classifier.predict(sc.transform([[r,t,p,a,s]]))[0]

        if(ans==0):
            print("NO BREAST CANCER.")
        else:
            print("BREAST CANCER.")
    elif(x==2):
        print("EXITING PROGRAM.")
        break
    else:
        print("Please enter a valid option.")







