# from motor.motor_asyncio import AsyncIOMotorClient
# from pymongo.server_api import ServerApi

# uri = "mongodb+srv://NovaSeele:Anhdungbadao%40262003@batoreach.39u4i.mongodb.net/"

# uri = "mongodb://localhost:27017/"

# mongodb+srv://NovaSeele:Anhdungbadao%40262003@batoreach.39u4i.mongodb.net/?retryWrites=true&w=majority&appName=BatoReach

# mongodb+srv://NovaSeele:Anhdungbadao%40262003@batoreach.39u4i.mongodb.net/ 

# Create a new client and connect to the server
from pymongo import MongoClient

# Kết nối MongoDB với pymongo
uri = "mongodb://localhost:27017/"
client = MongoClient(uri)

def get_mongodb_client():
    return client


