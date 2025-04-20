from pydantic import BaseModel
from typing import Optional, List

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: str
    full_name: str
    avatar: Optional[str] = None
    msv: Optional[str] = None


class UserUpdateAvatar(User):
    avatar: Optional[str] = None    
    

class UserMSVUpdate(BaseModel):
    msv: Optional[str] = None


class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str
    msv: Optional[str] = None


class UserInDB(User):
    password: str
    hashed_password: str


class UserChangePassword(BaseModel):
    old_password: str
    new_password: str



    



