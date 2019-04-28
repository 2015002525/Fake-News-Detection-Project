import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn import linear_model
import numpy as np

from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle
from sklearn.metrics.classification import classification_report, recall_score
from Pre_processing import *




def Test():
    trainData, trainClass, testData, testClass = load_data_Doc2vic()
    trainData = np.array(trainData)
    trainClass = np.array(trainClass)
    testData = np.array(testData)
    testClass = np.array(testClass)
    classifier = svm.SVC(kernel="rbf", verbose=2)
    classifier.fit(trainData, trainClass)
    testPrediction = classifier.predict(testData)
    return testClass, testPrediction


y_test,predictions = Test()
print(accuracy_score(y_test, predictions))
print(f1_score(y_test, predictions, average='weight'))
print(precision_score(y_test, predictions, average='weighted'))
print(recall_score(y_test, predictions, average='weighted'))




