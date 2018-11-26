#!/usr/bin/python

# import the necessary packages
import argparse
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric
from geoalchemy2 import Geometry
import numpy as np
from database import *
from preprocessing import *

Base = declarative_base()

class poiGeoData_Marousi(Base):
	__tablename__ = 'poiGeoData_Marousi'
	
	id = Column(Integer, primary_key=True)
	geom = Column(Geometry('POLYGON'))
	userid = Column(Integer)
	class_code = Column(String(10))
	theme = Column(String(254))
	class_name = Column(String(254))
	subclass_n = Column(String(254))
	name_l = Column(String(254))
	name_u = Column(String(254))
	x = Column(Numeric)
	y = Column(Numeric)
	
class edges(Base):
	__tablename__ = 'edges'
	
	id = Column(Integer, primary_key=True)
	geom = Column(Geometry('POLYGON'))
	access = Column(String(80))
	bridge = Column(String(80))
	from2 = Column(String(80))
	highway = Column(String(80))
	junction = Column(String(80))
	key = Column(String(80))
	lanes = Column(String(80))
	length = Column(String(80))
	maxspeed = Column(String(80))
	name = Column(String(80))
	oneway = Column(String(80))
	osmid = Column(String(165))
	ref = Column(String(80))
	service = Column(String(80))
	to = Column(String(80))
	tunnel = Column(String(80))
	width = Column(String(80))
	
def get_poi_id_to_closest_street_id_dict(session):

	"""
	*** This function maps each poi to its closest road id.
	***
	*** Returns - a dictionary consisting of the poi ids as
	*** 		  its keys and their corresponding closest 
	***			  road id as its value.
	"""
	
	from geoalchemy2 import functions
	from sqlalchemy import func
	
	# get the pois
	query = session.query(poiGeoData_Marousi)
	poi_ids = []
	for poi in query:
		poi_ids.append(poi.id)
	
	# construct a dictionary from their ids
	# also get its class_code
	poi_id_to_edge_id_dict = dict.fromkeys(poi_ids)
	for poi in query:
		poi_id_to_edge_id_dict[poi.id] = [int(poi.class_code), 0]
	
	# get the roads
	query2 = session.query(edges)
	
	# for each poi
	for poi in query:
		min_distance = 10000000000
		# for each edge
		for edge in query2:
			# compute its distance with respect to the poi
			distance = session.query(func.ST_Distance(func.ST_Transform(poi.geom, 32634), edge.geom))
			# if its distance is smaller than the current minimum
			if distance[0][0] < min_distance:
				# update the minimum distance and map the
				# road's id to the poi's id
				min_distance = distance[0][0]
				poi_id_to_edge_id_dict[poi.id][1] = edge.id
					
	return poi_id_to_edge_id_dict
	
def get_street_id_to_closest_pois_boolean_and_counts_per_label_dict(session, threshold):
	
	"""
	*** This function maps the street ids to their closest pois' label
	*** booleans and counts. The closest pois are determined by examining
	*** whether they are located within threshold distance of the street
	"""
	
	from geoalchemy2 import functions
	from sqlalchemy import func
	
	# get all the pois
	query = session.query(poiGeoData_Marousi)
	
	poi_ids = []
	for poi in  query:
		poi_ids.append(poi.id)
	
	# create a dictionary with the poi ids as its keys
	id_dict = dict.fromkeys(poi_ids)
	for poi in query:
		id_dict[poi.id] = [int(poi.class_code), 0, 0]
	
	# get the class codes set and encode the class codes to labels
	class_codes_set = get_class_codes_set()
	id_to_encoded_labels_dict, encoded_labels_set = get_encoded_labels(class_codes_set, id_dict)
	num_of_labels = len(encoded_labels_set)
	
	# get the streets
	query2 = session.query(edges)
	
	# create a dictionary with the street ids as its keys
	street_ids = []
	for edge in query2:
		street_ids.append(edge.id)
	street_id_to_closest_pois_boolean_and_counts_per_label_dict = dict.fromkeys(street_ids)
	
	# prepare the street id dictionary to be able to store the
	# boolean and count duplet for each of the class labels
	for edge in query2:
		street_id_to_closest_pois_boolean_and_counts_per_label_dict[edge.id] = [[0,0] for _ in range(0, num_of_labels)]
	
	# for each edge
	for edge in query2:
		# for each poi
		for poi in query:
			# within threshold distance
			distance = session.query(func.ST_Distance(func.ST_Transform(poi.geom, 32634), edge.geom))
			if distance[0][0] < threshold:
				# update the boolean and count values
				street_id_to_closest_pois_boolean_and_counts_per_label_dict[edge.id][encoded_labels_id_dict[poi.id][0][0]][0] = 1
				street_id_to_closest_pois_boolean_and_counts_per_label_dict[edge.id][encoded_labels_id_dict[poi.id][0][0]][1] += 1
				
	return street_id_to_closest_pois_boolean_and_counts_per_label_dict
	
def update_poi_id_dictionary(poi_id_to_street_id_dict, street_id_to_closest_pois_boolean_and_counts_per_label_dict):
	
	"""
	*** This function just copies the contents of street_id_to_closest_pois_dict
	*** to the newly created poi_id_to_closest_pois_boolean_count_dict based on
	*** the poi_id_to_street_id_dict values which will act as keys for the 
	*** street_id_to_closest_pois_dict dictionary
	"""
	
	# construct a dictionary from the ids of the pois
	poi_id_to_closest_pois_boolean_count_dict = dict.fromkeys(poi_id_to_street_id_dict.keys(),[])
	
	# for each poi id, get its closest road id and then copy
	# this road's closest poi label boolean and count to the
	# dictionary poi_id_to_closest_pois_boolean_count_dict
	for poi_id in poi_id_to_street_id_dict:
		poi_id_to_closest_pois_boolean_count_dict[poi_id] = street_id_to_closest_pois_boolean_and_counts_per_label_dict[poi_id_to_street_id_dict[poi_id]]
		
	return poi_id_to_closest_pois_boolean_count_dict
	
	
def get_closest_pois_boolean_and_counts_per_label_streets(session, threshold = 0):
	
	# get the dictionary mapping each poi id to that of its closest road
	poi_id_to_closest_street_id_dict = get_poi_id_to_closest_street_id_dict(session)
	
	# for every street id get the label boolean and counts values of the
	# pois located within threshold distance from it
	# (this will resemble the get_poi_id_to_boolean_and_counts_per_class_dict
	#  function but with road ids as the keys of the dictionary)
	threshold = 0
	street_id_to_label_boolean_counts_dict = get_street_id_to_closest_pois_boolean_and_counts_per_label_dict(session, threshold)
	
	# construct a dictionary similar to the one returned by get_poi_id_to_boolean_and_counts_per_class_dict
	# which will map a poi's id to the label boolean and count values of the poi's situated within threshold 
	# distance of the poi's closest road
	poi_id_to_closest_pois_boolean_count_dict = update_poi_id_dictionary(poi_id_to_closest_street_id_dict, street_id_to_label_boolean_counts_dict)
	return poi_id_to_closest_pois_boolean_count_dict
	
	
def get_class_codes_set():
	
	"""
	*** This function is responsible for reading the excel file
	*** containing the dataset labels (here stored in a more code-like
	*** manner rather than resembling labels).
	***
	*** Returns - a list of the class codes
	"""
	import pandas as pd
	
	# read the file containing the class codes
	df = pd.read_excel('/home/nikos/Desktop/Datasets/GeoData_PoiMarousi/GeoData_poiClasses.xlsx', sheet_name=None)
	
	# store the class codes (labels) in the list
	class_codes = list(df['poiClasses']['CLASS_CODE'])
	return class_codes
	
def get_poi_id_to_encoded_labels_dict(labels_set, id_dict):
	
	"""
	*** This function encodes our labels to values between 0 and len(labels_set)
	*** in order to have a more compact and user-friendly encoding of them.
	***
	*** Arguments - labels_set: the set of the labels (class codes) as we
	*** 			extracted them from the excel file
	***				id_dict: the dictionary containing the ids of the pois
	***
	*** Returns -	id_dict: an updated version of our pois dictionary
	***						 now mapping their ids to their encoded labels
	***				labels_set: the encoded labels set
	"""
	
	from sklearn.preprocessing import LabelEncoder
	
	# fit the label encoder to our labels set
	le = LabelEncoder()
	le.fit(labels_set)
	
	# map each poi id to its respective decoded label
	for key in id_dict:
		id_dict[key][0] = le.transform([id_dict[key][0]])
	
	return id_dict, le.transform(labels_set)
	
def get_poi_id_to_class_code_coordinates_dict(session):
	
	"""
	*** This function returns a dictionary with poi ids as its keys and a 
	*** list in the form of [< poi's class code >, < x coordinate > < y coordinate >]
	*** as its values.
	"""
		
	query = session.query(poiGeoData_Marousi)
	poi_ids = []
	for poi in  query:
		poi_ids.append(poi.id)
		
	poi_id_to_class_code_coordinates_dict = dict.fromkeys(poi_ids)
	for poi in query:
		poi_id_to_class_code_coordinates_dict[poi.id] = [int(poi.class_code), float(poi.x), float(poi.y)]
	
	return poi_id_to_class_code_coordinates_dict
	
def get_poi_id_to_boolean_and_counts_per_class_dict(session, num_of_labels, id_to_encoded_labels_dict, threshold):
	
	"""
	*** This function is responsible for mapping the pois to a list of two-element lists.
	*** The first element of that list will contain a  boolean value referring
	*** to whether a poi of that index's label is within threshold distance
	*** of the poi whose id is the key of this list in the dictionary. The second
	*** element contains the respective count of the pois belonging to the
	*** specific index's label that are within threshold distance of the poi-key.
	***
	*** For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	*** are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	*** id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	***
	*** Arguments - num_of_labels: the total number of the different labels
	*** 			encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	***				threshold: the aforementioned threshold
	"""
	
	from sklearn.neighbors import DistanceMetric
	from scipy.spatial import distance
	
	# define the apropriate distance metric for measuring distance between two pois
	dist = DistanceMetric.get_metric('euclidean')
	
	# get the poi ids and construct a dictionary with their ids as its keys
	query = session.query(poiGeoData_Marousi)
	poi_ids = []
	for poi in  query:
		poi_ids.append(poi.id)
		
	poi_id_to_label_boolean_counts_dict = dict.fromkeys(poi_ids)
	
	# add dummy values to the dictionary in order to initialize it
	# in a form that resembles its desired final form
	for poi in query:
		poi_id_to_label_boolean_counts_dict[poi.id] = [[0,0] for _ in range(0, num_of_labels)]
	
	# for every two different pois
	for id in poi_id_to_encoded_labels_dict:
		for id2 in poi_id_to_encoded_labels_dict:
			if id != id2:
				# get their coordinates
				point1 = (poi_id_to_encoded_labels_dict[id][1], poi_id_to_encoded_labels_dict[id][2])
				point2 = (poi_id_to_encoded_labels_dict[id2][1], poi_id_to_encoded_labels_dict[id2][2])
				# if the two points are within treshold distance, 
				# update the dictionary accordingly
				if distance.euclidean(point1, point2) < threshold:
					poi_id_to_label_boolean_counts_dict[id][poi_id_to_encoded_labels_dict[id2][0][0]][0] = 1
					poi_id_to_label_boolean_counts_dict[id][poi_id_to_encoded_labels_dict[id2][0][0]][1] += 1
	
	return poi_id_to_label_boolean_counts_dict
	
def get_closest_pois_boolean_and_counts_per_label(session, threshold = 0):
	
	"""
	*** This function returns a dictionary with the poi ids as its keys
	*** and two lists for each key. The first list contains boolean values
	*** dictating whether a poi of that index's label is within threshold
	*** distance with the key poi. The second list contains the counts of
	*** the pois belonging to the same index's label.
	
	*** Arguments - threshold: we only examine pois the distance between 
	*** 			which is below the given threshold
	"""
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict(session)
	
	# we read the different labels
	class_codes_set = get_class_codes_set()
	
	# we encode them so we can have a more compact representation of them
	poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict(class_codes_set, poi_id_to_class_code_coordinates_dict)
	
	return get_poi_id_to_boolean_and_counts_per_class_dict(session, len(encoded_labels_set), poi_id_to_encoded_labels_dict, threshold)

def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-db_name", "--db_name", required=True,
		help="database name")
	ap.add_argument("-usr_name", "--usr_name", required=True,
		help="name of the user")
	ap.add_argument("-usr_pswd", "--usr_pswd", required=True,
		help="password of the user")
	args = vars(ap.parse_args())
	
	# call the appropriate function to connect to the database
	engine = connect_to_db(args["db_name"], args["usr_name"], args["usr_pswd"])
	
	# create a session to start querying the database
	session = create_session(engine)
	
	#threshold = 1000.0
	#closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label(session, threshold)
	
	#closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(session)#, threshold)
	
if __name__ == "__main__":
   main()
