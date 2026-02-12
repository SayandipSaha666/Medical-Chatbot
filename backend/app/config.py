from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    pinecone_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    huggingfacehub_access_token: Optional[str] = None
    port: Optional[int] = None
    database_username: Optional[str] = None
    database_password: Optional[str] = None
    database_hostname: Optional[str] = None
    database_port: Optional[int] = None
    database_name: Optional[str] = None
    sqlalchemy_database_url: Optional[str] = None
    secret_key: Optional[str] = None
    algorithm: Optional[str] = None
    access_token_expire_minutes: Optional[int] = None
    clerk_jwks_url: Optional[str] = None
    clerk_issuer: Optional[str] = None
    imagekit_url_endpoint: Optional[str] = None
    imagekit_public_key: Optional[str] = None
    imagekit_private_key: Optional[str] = None

    class Config:
        env_file = ".env"

settings = Settings()
