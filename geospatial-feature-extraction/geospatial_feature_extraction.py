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

class Dian(Base):
	__tablename__ = 'dian'
	
	id = Column(Integer, primary_key=True)
	geom = Column(Geometry('POLYGON'))
	objectid = Column(Integer)
	unique_id = Column(String(50))
	worktype = Column(String(50))
	year_ = Column(String(50))
	parcelcode = Column(String(50))
	notes = Column(String(200))
	status_p = Column(String(254))
	location = Column(String(254))
	area_doc = Column(Integer)
	name = Column(String(50))
	area_diff = Column(Numeric)
	settlement = Column(String(50))
	ot = Column(String(50))
	shape_leng = Column(Numeric)
	shape_area = Column(Numeric)
	parcel_typ = Column(String(50))
	area_doc_1 = Column(Numeric)
	notes_1 = Column(String(254))
	origin = Column(String(254))
	first_name = Column(String(50))
	last_name = Column(String(50))
	f_name = Column(String(50))
	h_name = Column(String(50))
	m_n = Column(String(50))
	idio_klir = Column(String(20))

class Lpis(Base):
	__tablename__ = 'lpis2017'
	id = Column(Integer, primary_key=True)
	geom = Column(Geometry('POLYGON'))
	gid = Column(Integer)
	afm = Column(String(9))	
	lastname = Column(String(100))
	firstname = Column(String(40))
	fathername = Column(String(40))
	name = Column(String(100))
	eda_kodiko = Column(Integer)
	neoxartoyp = Column(String(13))
	kalkoin_na = Column(String(128))
	synidiofla = Column(Integer)
	synidioper = Column(Integer)
	area_ph = Column(Numeric)
	d_a = Column(Numeric)
	perim_p = Column(Numeric)
	afmidiokth = Column(String(9))
	nameidiokt = Column(String(50))
	iemtype = Column(Integer)
	prosopotyp = Column(Integer)
	shape_leng = Column(Numeric)
	shape_le_1 = Column(Numeric)
	shape_area = Column(Numeric)
	ota = Column(String(5))

	
class Mitroa(Base):
	__tablename__ = 'mitroa'
	id = Column(Integer, primary_key=True)
	geom = Column(Geometry('POLYGON'))
	gid = Column(Integer)
	objectid = Column(Integer)
	cd_tem = Column(String(17))
	temax_id = Column(String(9))
	parag_id = Column(String(9))
	type_tem = Column(String(1))
	xarto_kod = Column(String(13))
	aa_tem = Column(String(3))
	area_m = Column(String(6))
	perim = Column(String(6))
	ota_kodiko = Column(String(8))
	olv_count = Column(String(8))
	ae = Column(String(50))
	parag_id_1 = Column(Numeric)
	armh = Column(String(9))
	type = Column(Numeric)
	dge = Column(String(6))
	koi = Column(String(8))
	epon = Column(String(30))
	onom = Column(String(30))
	onompat = Column(String(20))
	etge = Column(String(4))
	artay = Column(String(10))
	afm = Column(String(9))
	notes = Column(String(254))
	stat = Column(Numeric)
	elper = Column(String(10))
	elegxos = Column(String(100))
	eldad = Column(Numeric)
	sumeld = Column(Numeric)
	sumelm = Column(Numeric)
	anoxhel = Column(Numeric)
	symfonh = Column(String(1))
	dhmos_kod = Column(String(8))
	shape_leng = Column(Numeric)
	shape_area = Column(Numeric)
	ota = Column(String(5))

def get_area_mxm(session, table_name):
	
	# this function returns the area (in sq. metres) of the rows of a given table
	
	from sqlalchemy import func
	
	assert(table_name in ["Mitroa", "Dian", "Lpis"])
	
	if table_name == "Mitroa":
		query_results = session.query(func.ST_Area(Mitroa.geom))
	elif table_name == "Dian":
		query_results = session.query(func.ST_Area(Dian.geom))
	else:
		query_results = session.query(func.ST_Area(Lpis.geom))
		
	return query_results
	
def get_perimeter(session, table_name):
	
	# this function returns the perimeter of the rows of a given table
	
	from sqlalchemy import func
	
	assert(table_name in ["Mitroa", "Dian", "Lpis"])
	
	if table_name == "Mitroa":
		query_results = session.query(func.ST_Perimeter(Mitroa.geom))
	elif table_name == "Dian":
		query_results = session.query(func.ST_Perimeter(Dian.geom))
	else:
		query_results = session.query(func.ST_Perimeter(Lpis.geom))
		
	return query_results
	
def get_vertices(session, table_name):
	
	# this function returns the number of vertices of the rows of a given table
	
	from sqlalchemy import func
	
	assert(table_name in ["Mitroa", "Dian", "Lpis"])
	
	if table_name == "Mitroa":
		query_results = session.query(func.ST_NPoints(Mitroa.geom))
	elif table_name == "Dian":
		query_results = session.query(func.ST_NPoints(Dian.geom))
	else:
		query_results = session.query(func.ST_NPoints(Lpis.geom))
		
	return query_results
	
def get_touches(session, table_name):
	
	# returns a dictionary which contains as keys the id of each geometry object
	# and holds two values: a boolean value stating whether this particular object
	# touches with another object and a numeric one referint to how many distinct objects it touches
	
	from sqlalchemy import func
	from geoalchemy2 import functions
	
	assert(table_name in ["Mitroa", "Dian", "Lpis"])
	
	if table_name == "Mitroa":
		query_results = session.query(Mitroa.id, Mitroa.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_Touches(geometry1, geometry2))
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1		
					
	elif table_name == "Dian":
		query_results = session.query(Dian.id, Dian.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_Touches(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
	else:
		query_results = session.query(Lpis.id, Lpis.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_Touches(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
		
	return id_dictionary
	
def get_instersects(session, table_name):
	
	# returns a dictionary which contains as keys the id of each geometry object
	# and holds two values: a boolean value stating whether this particular object
	# touches with another object and a numeric one referint to how many distinct objects it touches
	
	from sqlalchemy import func
	from geoalchemy2 import functions
	
	assert(table_name in ["Mitroa", "Dian", "Lpis"])
	
	if table_name == "Mitroa":
		query_results = session.query(Mitroa.id, Mitroa.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_Intersects(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
			
	elif table_name == "Dian":
		query_results = session.query(Dian.id, Dian.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_Intersects(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
	else:
		query_results = session.query(Lpis.id, Lpis.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_Intersects(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
		
	return id_dictionary
	
def get_covers(session, table_name):
	
	# returns a dictionary which contains as keys the id of each geometry object
	# and holds two values: a boolean value stating whether this particular object
	# touches with another object and a numeric one referint to how many distinct objects it touches
	
	from sqlalchemy import func
	from geoalchemy2 import functions
	
	assert(table_name in ["Mitroa", "Dian", "Lpis"])
	
	if table_name == "Mitroa":
		query_results = session.query(Mitroa.id, Mitroa.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_Covers(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
			
	elif table_name == "Dian":
		query_results = session.query(Dian.id, Dian.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_Covers(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
	else:
		query_results = session.query(Lpis.id, Lpis.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_Covers(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
		
	return id_dictionary
	
def get_coveredbys(session, table_name):
	
	# returns a dictionary which contains as keys the id of each geometry object
	# and holds two values: a boolean value stating whether this particular object
	# touches with another object and a numeric one referint to how many distinct objects it touches
	
	from sqlalchemy import func
	from geoalchemy2 import functions
	
	assert(table_name in ["Mitroa", "Dian", "Lpis"])
	
	if table_name == "Mitroa":
		query_results = session.query(Mitroa.id, Mitroa.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_CoveredBy(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
			
	elif table_name == "Dian":
		query_results = session.query(Dian.id, Dian.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_CoveredBy(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
	else:
		query_results = session.query(Lpis.id, Lpis.geom)
		id_list = []
		for row in query_results:
			id_list.append(row[0])
		id_dictionary = dict.fromkeys(id_list)
		
		for key in id_dictionary:
			id_dictionary[key] = [0, 0]
		
		for id1, geometry1 in query_results:
			#print(id1, geometry1)
			for id2, geometry2 in query_results:
				query_results2 = session.query(func.ST_CoveredBy(geometry1, geometry2))
				
				#print(query_results2)
				if query_results2[0][0] == True:
					id_dictionary[id1][0], id_dictionary[id1][1] == 1, id_dictionary[id1][1] + 1
					id_dictionary[id2][0], id_dictionary[id2][1] == 1, id_dictionary[id2][1] + 1
		
	return id_dictionary
	
def get_statistics_for_edges(session, table_name):
	
	# this function returns the mean and variance of the number of edges of the rows of a given table
	
	from sqlalchemy import func
	
	assert(table_name in ["Mitroa", "Dian", "Lpis"])
	
	vertices = get_vertices(session, table_name)
	
	vertices = np.asarray([float(vertex[0]) for vertex in vertices])
	
	return np.mean(vertices), np.var(vertices)
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-db_name", "--db_name", required=True,
		help="database name")
	ap.add_argument("-usr_name", "--usr_name", required=True,
		help="name of the user")
	ap.add_argument("-usr_pswd", "--usr_pswd", required=True,
		help="password of the user")
	ap.add_argument("-tbl_name", "--tbl_name", required=True,
		help="table name")
	args = vars(ap.parse_args())
	
	# call the appropriate function to connect to the database
	engine = connect_to_db(args["db_name"], args["usr_name"], args["usr_pswd"])
	
	CustomTableClass = get_custom_table_class(args["tbl_name"], engine)
	
	# create a session to start querying the database
	session = create_session(engine)
	
	# A simple query to try the connection
	query = session.query(CustomTableClass)
	
	for mitroo in  query:
		print(mitroo.gid)
		
	return
	
	# Trying GIS queries
	
	areas_mitroa = get_area_mxm(session, "Mitroa")
	areas_list = []
	for area in areas_mitroa:
		areas_list.append(area[0])
	areas = np.asarray(areas_list)
	
	perimeters_mitroa = get_perimeter(session, "Mitroa")
	perimeters_list = []
	for perimeter in perimeters_mitroa:
		perimeters_list.append(perimeter[0])
	perimeters = np.asarray(perimeters_list)
	
	vertices_mitroa = get_perimeter(session, "Mitroa")
	vertices_list = []
	for vertex in vertices_mitroa:
		vertices_list.append(vertex[0])
	vertices = np.asarray(vertices_list)
	
	#mean_edges, variance_edges = get_statistics_for_edges(session, "Mitroa")
	#print(mean_edges, variance_edges)
	
	touches = get_touches(session, "Mitroa")
	print(touches)
	#for touch in touches:
	#	print(touch)
	
	X_train, X_test = standardize_data()

if __name__ == "__main__":
   main()
