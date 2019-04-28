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
from sklearn.metrics.classification import classification_report, accuracy_score, precision_score, f1_score, \
    recall_score


def load_data_Doc2vic():
    model = Doc2Vec.load('Files/FakeModel.d2v')

    fHam = open('real', 'r')
    fSpam = open('fake', 'r')

    hamArray = []
    spamArray = []
    for line in fHam:
        hamArray.append(line[:-1])
    for line in fSpam:
        spamArray.append(line[:-1])
    shuffle(hamArray)
    shuffle(spamArray)

    trainData = []
    testData = []
    trainClass = []
    testClass = []

    for i in range(0, int(len(hamArray) * 0.8)):
        trainData.append(model.docvecs['real_' + str(i)])
        trainClass.append(0)

    for i in range(int(len(hamArray) * 0.8), len(hamArray)):
        testData.append(model.docvecs['real_' + str(i)])
        testClass.append(0)

    for i in range(0, int(len(spamArray) * 0.8)):
        trainData.append(model.docvecs['fake_' + str(i)])
        trainClass.append(1)

    for i in range(int(len(spamArray) * 0.8), len(spamArray)):
        testData.append(model.docvecs['fake_' + str(i)])
        testClass.append(1)

    return trainData,trainClass,testData,testClass





