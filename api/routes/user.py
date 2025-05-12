from fastapi import APIRouter, Depends, UploadFile, File, Form
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from schemas.user import (
    User,
    UserInDB,
    Token,
    UserCreate,
    UserUpdateAvatar,
    UserChangePassword,
    UserMSVUpdate,
    UserRole,
)
from db.repositories.user import get_current_user, get_lecturer_user
from services.user import (
    upload_avatar_service,
    register_user_service,
    login_for_access_token_service,
    get_current_user_service,
    change_password_service,
    add_msv_service,
    set_user_role_service,
)

router = APIRouter()


# Upload Avatar
@router.post("/upload-avatar", response_model=UserUpdateAvatar)
def upload_avatar(
    avatar: UploadFile = File(...), current_user: UserInDB = Depends(get_current_user)
):
    return upload_avatar_service(avatar, current_user)


# Route to update MSV - only for students
@router.post("/add-msv", response_model=User)
def add_msv(msv_data: UserMSVUpdate, current_user: User = Depends(get_current_user)):
    return add_msv_service(current_user, msv_data.msv)


# Register a new user
@router.post("/register", response_model=UserCreate)
def register_user(user: UserCreate):
    return register_user_service(user)


# Modified login request model to include role
class LoginRequest(BaseModel):
    username: str
    password: str
    role: str = "student"  # Default to student if not provided


# Login and get access token ( test route for FastAPI Docs )
@router.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), role: str = Form("student")
):
    return login_for_access_token_service(form_data, role)


# Login and get access token
@router.post("/login", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), role: str = Form("student")
):
    return login_for_access_token_service(form_data, role)


# Get current user
@router.get("/users/me", response_model=User)
def read_users_me(current_user: User = Depends(get_current_user)):
    return get_current_user_service(current_user)


# Change user password
@router.put("/change-password", response_model=UserInDB)
def change_password(
    user: UserChangePassword, current_user: UserInDB = Depends(get_current_user)
):
    return change_password_service(user, current_user)


# Model for role change request
class RoleUpdateRequest(BaseModel):
    username: str
    role: UserRole


# Set user role - only lecturers can do this
@router.put("/set-role", response_model=User)
def set_user_role(
    role_data: RoleUpdateRequest, current_user: UserInDB = Depends(get_lecturer_user)
):
    """Set a user's role - only lecturers can perform this action"""
    return set_user_role_service(role_data.username, role_data.role, current_user)


# Check if current user is a lecturer
@router.get("/is-lecturer")
def check_if_lecturer(current_user: UserInDB = Depends(get_current_user)):
    """Check if the current user is a lecturer"""
    return {"is_lecturer": current_user.role == UserRole.LECTURER}


# Check if current user is a guest
@router.get("/is-guest")
def check_if_guest(current_user: UserInDB = Depends(get_current_user)):
    """Check if the current user is a guest"""
    return {"is_guest": current_user.role == "guest"}
