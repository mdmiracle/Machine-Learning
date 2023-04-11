import numpy as np
import pandas as pd
import sklearn


df= pd.read_csv("C:\\Users\\91779\\Downloads\\archive.zip")
df.drop(['Name','SibSp','Parch',"Ticket","Cabin","Embarked",'PassengerId'],axis=1,inplace=True)
#print(df.head())
target=df.Survived
input=df.drop('Survived',axis=1)
#print(target)
dummies=pd.get_dummies(input.Sex)
#print(dummies.head(3))
#df['female'],df['male']=dummies.female, dummies.male
input=pd.concat([input,dummies],axis=1)
input.drop('Sex',axis=1,inplace=True)
#print(input.head())
#print(input.columns[input.isna().any()])
#print(input.Age[0:10])
input.Age=input.Age.fillna(input.Age.mean())
input.Fare=input.Fare.fillna(input.Fare.mean())
#print(input.columns[input.isna().any()])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input,target,test_size=0.2)

print(len(x_train), len(x_test), len(y_train), len(y_test))
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(x_test[:10],y_test[:10])
print(model.predict(x_test[:10]))
print(model.predict_proba(x_test[:10]))
print(dummies.head(3))

