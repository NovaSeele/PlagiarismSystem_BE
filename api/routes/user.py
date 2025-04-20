from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm

from schemas.user import User, UserInDB, Token, UserCreate, UserUpdateAvatar, UserChangePassword, UserMSVUpdate
from db.repositories.user import get_current_user
from services.user import (
    upload_avatar_service,
    register_user_service,
    login_for_access_token_service,
    get_current_user_service,
    change_password_service,
    add_msv_service
)

router = APIRouter()

# Upload Avatar
@router.post("/upload-avatar", response_model=UserUpdateAvatar)
def upload_avatar(avatar: UploadFile = File(...), current_user: UserInDB = Depends(get_current_user)):
    return upload_avatar_service(avatar, current_user)

# Route to update MSV
@router.post("/add-msv", response_model=User)
def add_msv(msv_data: UserMSVUpdate, current_user: User = Depends(get_current_user)):
    return add_msv_service(current_user, msv_data.msv)

# Register a new user
@router.post("/register", response_model=UserCreate)
def register_user(user: UserCreate):
    return register_user_service(user)

# Login and get access token ( test route for FastAPI Docs )
@router.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    return login_for_access_token_service(form_data)

# Login and get access token
@router.post("/login", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    return login_for_access_token_service(form_data)

# Get current user
@router.get("/users/me", response_model=User)
def read_users_me(current_user: User = Depends(get_current_user)):
    return get_current_user_service(current_user)

# Change user password
@router.put("/change-password", response_model=UserInDB)
def change_password(user: UserChangePassword, current_user: UserInDB = Depends(get_current_user)):
    return change_password_service(user, current_user)
