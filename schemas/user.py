from pydantic import BaseModel
from typing import Optional, List, Literal
from enum import Enum


class UserRole(str, Enum):
    STUDENT = "student"
    LECTURER = "lecturer"
    GUEST = "guest"


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class BaseUser(BaseModel):
    """Base model for all users"""

    username: str
    email: str
    full_name: str
    avatar: Optional[str] = None


class Student(BaseUser):
    """Student user with limited permissions"""

    role: UserRole = UserRole.STUDENT
    msv: Optional[str] = None


class Lecturer(Student):
    """Lecturer user with full permissions, inherits from Student"""

    role: UserRole = UserRole.LECTURER
    msv: Optional[str] = None  # Lecturers don't need msv but kept for inheritance


class User(Student):
    """Generic user model, backward compatible"""

    role: UserRole = UserRole.STUDENT


class UserUpdateAvatar(BaseUser):
    avatar: Optional[str] = None
    role: Optional[UserRole] = None


class UserMSVUpdate(BaseModel):
    msv: Optional[str] = None


class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str
    msv: Optional[str] = None
    role: UserRole = UserRole.STUDENT


class UserInDB(BaseUser):
    password: str
    hashed_password: str
    msv: Optional[str] = None
    role: UserRole = UserRole.STUDENT


class UserChangePassword(BaseModel):
    old_password: str
    new_password: str
