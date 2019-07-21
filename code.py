import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os
import warnings
import keras
from keras.models import Sequential 
from keras.layers import Dense 
import tensorflow as tf
warnings.filterwarnings('ignore') 
#plt.style.use('seaborn')
plt.style.use('fivethirtyeight')

data=pd.read_csv('1.csv')
print(data.head())

data.drop('Serial No.', axis=1, inplace=True)

data.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="rainbow")
plt.xlabel('Statistics',)
plt.ylabel('Value')
plt.title("General Statistics of Admissions")
plt.show()

data_sort=data.sort_values('Response ', ascending=False)


X=data_sort.iloc[:,0:8].values

y=data_sort.iloc[:,8].values



from sklearn.model_selection import train_test_split   #cross_validation doesnt work any more
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 

from sklearn.preprocessing import StandardScaler 
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

classifier_6=Sequential()
classifier_6.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=8))
classifier_6.add(Dense(output_dim=5,init='uniform',activation='relu'))
classifier_6.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier_6.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier_6.fit(X_train, y_train,batch_size=10,nb_epoch=100)

y_pred=classifier_6.predict(X_test)
y_pred=(y_pred>0.7)
print(y_pred)


from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(y_test,y_pred)
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()