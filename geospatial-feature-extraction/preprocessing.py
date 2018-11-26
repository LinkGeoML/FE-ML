#!/usr/bin/python

def standardize_data(X_train, X_test):
	from sklearn.preprocessing import StandardScaler
	
	standard_scaler = StandardScaler()
	X_train = standard_scaler.fit_transform(X_train)
	X_test = standard_scaler.transform(X_test)
	
	return X_train, X_test
