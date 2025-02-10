from pymongo import MongoClient
import pandas as pd
import json

#save the url path
url = "mongodb+srv://rakshit2004gupta:12345@cluster0.d6zu3.mongodb.net/?retryWrites=true&w=majority"

#create a client and connect to the server
client = MongoClient(url)

#create a database and colection name 

DATABASE_NAME = 'Sensor'
COLLECTION_NAME = 'waferfault'

df = pd.read_csv("C:\Users\HP\OneDrive\Desktop\SensorProject\notesbooks\wafer_23012020_041211.csv")

df = df.drop("Unnamed: 0", axis = 1)

json_record = list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)



