#!/usr/bin/python

def create_session(engine):
	from sqlalchemy.orm import sessionmaker
	
	Session = sessionmaker(bind=engine)
	session = Session()
	
	return session

def connect_to_db(db_name, usr_name, usr_pswd):
	
	from sqlalchemy import create_engine
	
	request = 'postgresql://{}:{}@localhost/{}'.format(usr_name, usr_pswd, db_name)
	engine = create_engine(request, echo=True)
	
	return engine

def get_custom_table_class(table_name, engine):
	
	from sqlalchemy import MetaData, Table
	from sqlalchemy.ext.automap import automap_base
	
	# produce our own MetaData object
	metadata = MetaData()

	# we can reflect it ourselves from a database, using options
	# such as 'only' to limit what tables we look at...
	metadata.reflect(engine, only=[table_name])
	
	# we can then produce a set of mappings from this MetaData.
	Base = automap_base(metadata=metadata)

	# calling prepare() just sets up mapped classes and relationships.
	Base.prepare()
	
	# mapped classes are ready
	exec("CustomTableClass = Base.classes.%s"%(table_name))
	
	return CustomTableClass
