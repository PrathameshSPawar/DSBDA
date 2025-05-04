import numpy as np
import pandas as pd
air=pd.read_csv('airquality.csv')
air.shape
air.head(10)
air.count()
air.isnull().sum()
air.describe()
air.info()
A=air.dropna()
A.head(10)
A=air.fillna(0)
A.shape
A.head()
A=air.fillna(method='pad')
A.head()
A = air.fillna(method='backfill')
A.head()
A=air['Ozone'].replace(np.nan,air['Ozone'].mean())
A=air['Ozone'].replace(np.nan,air['Ozone'].median())
A=air['Ozone'].replace(np.nan,air['Ozone'].mode())
A.head()

from sklearn.impute import SimpleImputer
imp= SimpleImputer(missing_values=np.nan,strategy='mean')
A=imp.fit_transform(air)
A
A=pd.DataFrame(A,columns=air.columns)
A.head()

from sklearn.model_selection import train_test_split
len(A)
train,test=train_test_split(A)
len(train)
len(test)
train.head()
train,test=train_test_split(A,test_size=0.20)
len(test)
len(train)
A.describe()


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
B=scaler.fit_transform(A)
pd.DataFrame(B).describe()

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
B=scaler.fit_transform(A)
B=pd.DataFrame(B).describe()
B
from sklearn.preprocessing import Binarizer
bin=Binarizer(threshold=0.5)
B=bin.fit_transform(B)
pd.DataFrame(B)

data=pd.read_csv('student.csv')
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
B=le.fit_transform(data['name'])
B
B=data[:]
B['name']=le.fit_transform(B['name'])
B
A

from sklearn.linear_model import LinearRegression
X=A['Ozone'].values
X=X.reshape(-1,1)
Y=A['Temp']
model=LinearRegression()
model.fit(X,Y)
model.score(X,Y)*100
model.predict([[128]])
import matplotlib.pyplot as plt
plt.scatter(X,Y)
