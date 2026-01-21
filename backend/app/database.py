from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,declarative_base
from .config import settings
from urllib.parse import quote

encoded_password = quote(settings.database_password)

SQLALCHEMY_DATABASE_URL = (
    f"postgresql://{settings.database_username}:{encoded_password}"
    f"@{settings.database_hostname}:{settings.database_port}/{settings.database_name}"
)

# SQLALCHEMY_DATABASE_URL=settings.sqlalchemy_database_url

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()