# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

le = preprocessing.LabelEncoder()

veri = pd.read_csv('C:/Users/ibrahim/Desktop/ödev1/CAR DETAILS FROM CAR DEKHO.csv')
veri=veri.drop(["seller_type"],axis=1)
car_name=veri.iloc[:,0:1]
veri=veri.drop(["name"],axis=1)
fuel=veri.iloc[:,3:4].values
trans=veri.iloc[:,4:5].values
owner=veri.iloc[:,-1].values
fuel= le.fit_transform(fuel)
trans= le.fit_transform(trans)
owner= le.fit_transform(owner)
fuel=pd.DataFrame(data=fuel,columns=["fuel"])
trans=pd.DataFrame(data=trans,columns=["transmissior"])
owner=pd.DataFrame(data=owner,columns=["owner"])
veri=veri.iloc[:,0:3]
veri= pd.concat([veri,fuel,trans,owner],axis=1)



x=veri.drop(["selling_price"],axis=1)
y=veri.iloc[:,1:2]
X=x.values
Y=y.values

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

#linear
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)



#polinom
poly_reg = preprocessing.PolynomialFeatures()
x_poly=poly_reg.fit_transform(X)

x_train_poly, x_test_poly, y_train, y_test = train_test_split(x_poly,Y,test_size=0.33,random_state=0)

lr2=LinearRegression()
lr2.fit(x_train_poly,y_train)
Y_pred_poly=lr2.predict(x_test_poly)

#StandardScale

sc = preprocessing.StandardScaler()
x_train_olc=sc.fit_transform(x_train)
x_test_olc=sc.fit_transform(x_test)
y_train_olc=sc.fit_transform(y_train)
y_test_olc=sc.fit_transform(y_test)

#svr 

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_train_olc,y_train_olc)
y_pred_svr = svr_reg.predict(x_test_olc)


#karar ağacı

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(x_train,y_train)
y_pred_dt = dtr.predict(x_test)

#grafik

plt.scatter(x[["km_driven"]].values, y.values, color='blue')
plt.plot(x_test[:,1], y_pred_lr,color='red')
plt.show()






