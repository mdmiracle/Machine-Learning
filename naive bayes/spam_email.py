import numpy as np
import pandas as pd
import sklearn 
df=pd.read_csv(r"C:\Users\91779\Downloads\archive (1).zip",encoding='ANSI')
print(df.head(4))
#print(df.groupby('v1').describe())
#print(len(df.axes[1]))
df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)
print(len(df.axes[1]))
print(df.groupby('v1').describe())

df['spam']=df.v1.apply(lambda x: 1 if x=='spam' else 0)
print(df.spam)
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(df.v2,df.spam,test_size=0.25)

#we have to convert our v2 column into numbers using count vectorizer technique
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
x_train_count=v.fit_transform(x_train.values)
print(x_train_count.toarray()[:3])

#using multinomial naive bayes because features have discrete values
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train_count,y_train)

email=[
    'hey we will play tomorrow',
    'you get 50% flat discount'
]

email_count= v.transform(email)
print(model.predict(email_count))

x_test_count=v.transform(x_test)
print(model.score(x_test_count,y_test))

#by the upper method we are first converting our text to vector of number by countvectorizer and than applying multinomial model, we can do this whole task with the help of pipeline at once
from sklearn.pipeline import Pipeline
clf= Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

ex=[
    'you must come to be awarded 50% flat discount on your purchace',
    'we will give you gifts for playing'
]
print(clf.predict(ex),clf.predict_proba(ex))

