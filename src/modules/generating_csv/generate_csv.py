from lib.databaseIO import pgIO

pgIO.getAllData(query="select * from films", dbName='dvdrental')