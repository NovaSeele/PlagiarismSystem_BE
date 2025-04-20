from typing import List
from fastapi import Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
import cloudinary
import cloudinary.uploader

from dependency.user import get_password_hash, create_access_token
from db.repositories.user import authenticate_user, get_current_user, get_user_collection
from schemas.user import User, UserInDB, Token, UserCreate, UserChangePassword

# Configuration       
cloudinary.config( 
    cloud_name = "dxovnpypb", 
    api_key = "163478744136852", 
    api_secret = "sjuU6l-A4wTGCHxwcYZ5HecB0xg", # Click 'View API Keys' above to copy your API secret
    secure = True
)

# Upload Avatar
def upload_avatar_service(avatar: UploadFile, current_user: UserInDB):
    user_collection = get_user_collection()

    # Upload the avatar file to Cloudinary
    upload_result = cloudinary.uploader.upload(avatar.file, public_id=f"{current_user.username}_avatar")
    avatar_url = upload_result.get("secure_url")

    # Update user's avatar URL in the database
    user_collection.update_one(
        {"username": current_user.username}, {"$set": {"avatar": avatar_url}}
    )

    # Fetch updated user details
    updated_user = user_collection.find_one({"username": current_user.username})
    return updated_user

# Register a new user
def register_user_service(user: UserCreate):
    user_collection = get_user_collection()

    # Check if the user already exists
    existing_user = user_collection.find_one(
        {"$or": [{"username": user.username}, {"email": user.email}]})
    if existing_user:
        raise HTTPException(
            status_code=400, detail="Username or email already registered")

    hashed_password = get_password_hash(user.password)
    user_dict = user.model_dump()
    user_dict['hashed_password'] = hashed_password

    # Insert the user into the database
    result = user_collection.insert_one(user_dict)

    if result.inserted_id:
        return user
    raise HTTPException(status_code=400, detail="Registration failed")

# Login and get access token
def login_for_access_token_service(form_data: OAuth2PasswordRequestForm):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username, email, or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Get current user
def get_current_user_service(current_user: User):
    return current_user

# Change user password
def change_password_service(user: UserChangePassword, current_user: UserInDB):
    user_collection = get_user_collection()

    if user.old_password == user.new_password:
        raise HTTPException(
            status_code=400, detail="New password cannot be the same as the old password")

    if not user.old_password == current_user.password:
        raise HTTPException(
            status_code=400, detail="Old password is incorrect")

    new_hashed_password = get_password_hash(user.new_password)

    # Update password in the database
    user_collection.update_one(
        {"username": current_user.username},
        {"$set": {
            "hashed_password": new_hashed_password,
            "password": user.new_password
        }}
    )

    # Fetch updated user information
    updated_user = user_collection.find_one({"username": current_user.username})
    return updated_user

# Update user's MSV
def add_msv_service(user: User, msv: str):
    user_collection = get_user_collection()

    # Update user's MSV in the database
    user_collection.update_one(
        {"username": user.username}, {"$set": {"msv": msv}}
    )

    # Fetch updated user details
    updated_user = user_collection.find_one({"username": user.username})
    
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Ensure `msv` is updated in the response
    if updated_user.get("msv") != msv:
        raise HTTPException(status_code=400, detail="Failed to update MSV")

    return User(**updated_user)