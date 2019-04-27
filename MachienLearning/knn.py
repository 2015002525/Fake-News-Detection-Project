# -*- coding: utf-8 -*-
"""
@author: EyadMShokry
"""
from sklearn.metrics import classification_report

from DataPreProcessor import DataPreProcessor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#Reading Data
dataProcessor = DataPreProcessor("C:\\Users\\Ahmed\\OneDrive\\Desktop\\Fake-News-Detection-master\\all.xlsx")
df = dataProcessor.LoadData()
print("dudud")

#Spliting data to training data (80%) & testing data (20%)
train_X, test_X, train_Y, test_Y = train_test_split(df['content'].fillna(' '), df['label'], test_size=0.05)

#Buliding Pipeline
kVals = []
accuracies = []
for k in range(1,2,2):
    print(k)
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('clf', KNeighborsClassifier(n_neighbors=k)),])
    text_clf = text_clf.fit(train_X, train_Y)
    #Prediction
    predection = text_clf.predict(test_X)
    score = text_clf.score(test_X, test_Y)
    kVals.append(k)    
    accuracies.append(score)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
#

i=np.argmax(accuracies)
knn = KNeighborsClassifier(n_neighbors=kVals[0])
print("123")
knn= knn.fit(train_X,train_Y)
print("46")
print('KNN score: %f' % knn.fit(train_X, train_Y).score(test_X, test_Y))

plt.title("Model's Accuracy comparing to K values")    
plt.plot(kVals, accuracies)
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.show()
