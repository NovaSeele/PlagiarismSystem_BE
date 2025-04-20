# from passlib.context import CryptContext
# from fastapi.security import OAuth2PasswordBearer
# from database import get_monogodb_client
# import motor.motor_asyncio
# from pymongo import database
# from gridfs import GridFSBucket  # Import GridFSBucket từ pymongo

# client = get_monogodb_client()
# db = client['NovaSeelePlagiarismChecker']

# # Tạo GridFSBucket đúng cách
# def get_fs():
#     return GridFSBucket(db)

# def get_collection(collection_name):
#     return db[collection_name]

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from database import get_mongodb_client
from gridfs import GridFSBucket

client = get_mongodb_client()
db = client['NovaSeelePlagiarismChecker']

# GridFSBucket cho việc lưu file
fs = GridFSBucket(db)

def get_collection(collection_name):
    return db[collection_name]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
