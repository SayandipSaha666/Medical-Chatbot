from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    pinecone_api_key: str
    groq_api_key: str
    huggingfacehub_access_token: str
    port: int = 8080
    database_username: str
    database_password: str
    database_hostname: str
    database_port: int
    database_name: str
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    sqlalchemy_database_url: Optional[str] = None
    clerk_jwks_url: str
    clerk_issuer: str
    imagekit_public_key: str
    imagekit_private_key: str
    imagekit_url_endpoint: str

    class Config:
        env_file = ".env"

settings = Settings()
