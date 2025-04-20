import pathlib
from pydantic_settings import BaseSettings

ROOT = pathlib.Path(__file__).parent.parent.parent

SECRET_KEY = "4ab5be85c8c56eecdd547f7831979be83de58a6768d10a314f54cda4e4d67ffe"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8888
    # DB_URL: str
    # DB_PASSWORD: str
    # DB_HOSTNAME: str
    # DB_PORT: int
    # DB_NAME: str
    # SECRET_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
