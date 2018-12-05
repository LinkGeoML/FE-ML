#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from pois_feature_extraction import *
from textual_feature_extraction import *
from feml import *
import nltk
import itertools
import random

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

np.random.seed(1234)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def default_parameters_single_fold(X_train, X_test, y_train, y_test):
    # preprocess it
	X_train, X_test = standardize_data(X_train, X_test)
	
	# declare classifiers
	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Naive Bayes"]

	classifiers = [
		KNeighborsClassifier(3),
		SVC(kernel="linear", C=0.025),
		SVC(gamma=2, C=1),
		DecisionTreeClassifier(max_depth=5),
		RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
		GaussianNB()]
	
	# iterate over classifiers and produce results
	for name, clf in zip(names, classifiers):
		clf.fit(X_train, y_train)
		score = clf.score(X_test, y_test)
		print("Accuracy Score of {0} classifier: {1}".format(name, score))
		
def default_parameters_5_fold(X_train, X_test, y_train, y_test):
	
	X = list(np.concatenate((X_train, X_test), axis=0))
	y = list(np.concatenate((y_train, y_test), axis=0))
	
	c = list(zip(X, y))

	random.shuffle(c)

	X, y = zip(*c)
	
	X = np.asarray(X)
	y = np.asarray(y)
	
	kf = KFold(n_splits=5)
	count = 1
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		print("Displaying results for fold {0}".format(count))
		
		names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Naive Bayes"]

		classifiers = [
			KNeighborsClassifier(3),
			SVC(kernel="linear", C=0.025),
			SVC(gamma=2, C=1),
			DecisionTreeClassifier(max_depth=5),
			RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
			GaussianNB()]
		
		# iterate over classifiers and produce results
		for name, clf in zip(names, classifiers):
			clf.fit(X_train, y_train)
			score = clf.score(X_test, y_test)
			print("Accuracy Score of {0} classifier: {1}".format(name, score))
			
		count += 1
		
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-pois_tbl_name", "--pois_tbl_name", required=True,
		help="name of table containing pois information")
	ap.add_argument("-roads_tbl_name", "--roads_tbl_name", required=True,
		help="name of table containing roads information")
	ap.add_argument("-threshold", "--threshold", required=True,
		help="threshold for distance-specific features")
	ap.add_argument("-k", "--k", required=True,
		help="the number of the desired top-k most frequent tokens")
	ap.add_argument("-n", "--n", required=True,
		help="the n-gram size")
	args = vars(ap.parse_args())
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	# get the data
	
	#poi_ids_train, poi_ids_test = get_train_test_poi_ids(conn, args)
	#X_train, y_train, X_test, y_test = get_train_test_sets(conn, args, poi_ids_train, poi_ids_test)
	
	# save it in csv files
	
	#np.savetxt("train.csv", X_train, delimiter=",")
	#np.savetxt("test.csv", X_test, delimiter=",")
	#np.savetxt("labels_train.csv", y_train, delimiter=",")
	#np.savetxt("labels_test.csv", y_test, delimiter=",")
	
	# load it from csv files
	X_train = np.genfromtxt('train.csv', delimiter=',')
	X_test = np.genfromtxt('test.csv', delimiter=',')
	y_train = np.genfromtxt('labels_train.csv', delimiter=',')
	y_test = np.genfromtxt('labels_test.csv', delimiter=',')
	
	#default_parameters_single_fold(X_train, X_test, y_train, y_test)
	
	default_parameters_5_fold(X_train, X_test, y_train, y_test)
	
if __name__ == "__main__":
   main()
