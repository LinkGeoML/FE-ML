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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
	
	poi_ids_train, poi_ids_test = get_train_test_poi_ids(conn, args)
	X_train, y_train, X_test, y_test = get_train_test_sets(conn, args, poi_ids_train, poi_ids_test)
	
	numpy.savetxt("train.csv", X_train, delimiter=",")
	numpy.savetxt("test.csv", X_train, delimiter=",")
	return
	
	X_train, X_test = standardize_data(X_train, X_test)
	
	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Naive Bayes"]

	classifiers = [
		KNeighborsClassifier(3),
		SVC(kernel="linear", C=0.025),
		SVC(gamma=2, C=1),
		DecisionTreeClassifier(max_depth=5),
		RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
		GaussianNB()]
		
	 # iterate over classifiers
	for name, clf in zip(names, classifiers):
		clf.fit(X_train, y_train)
		score = clf.score(X_test, y_test)
		
		print("Classification Score of {0} classifier: {1}".format(name, score))
		
		print("Printing relevant confusion matrices...")
		y_pred = clf.predict(y_test)
		
		# Compute confusion matrix
		cnf_matrix = confusion_matrix(y_test, y_pred)
		np.set_printoptions(precision=2)

		# Plot non-normalized confusion matrix
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=class_names,
							  title='Confusion matrix, without normalization')

		# Plot normalized confusion matrix
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
							  title='Normalized confusion matrix')

		plt.show()

	
if __name__ == "__main__":
   main()
