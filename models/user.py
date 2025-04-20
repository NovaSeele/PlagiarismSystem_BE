from typing import Optional

from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt

from config.config import SECRET_KEY, ALGORITHM
from db.session import get_collection
from dependency.user import verify_password, oauth2_scheme
from schemas.user import UserInDB, TokenData

def get_user_collection():
    user_collection = get_collection('users')
    return user_collection

def get_user_by_username_or_email(identifier: str) -> Optional[UserInDB]:
    user_collection = get_user_collection()
    user_data = user_collection.find_one(
        {"$or": [{"username": identifier}, {"email": identifier}]}
    )
    if user_data:
        return UserInDB(**user_data)
    return None

def authenticate_user(identifier: str, password: str) -> Optional[UserInDB]:
    user = get_user_by_username_or_email(identifier)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user_by_username_or_email(token_data.username)
    if user is None:
        raise credentials_exception
    return user
