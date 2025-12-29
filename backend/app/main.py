# For documentation head over to http://127.0.0.1:8000/redoc or http://127.0.0.1:8000/docs route

from fastapi import FastAPI
from . import models
from .database import engine,get_db
from fastapi.middleware.cors import CORSMiddleware
from .routers import user,chats,messages,images
from dotenv import load_dotenv
from fastapi import Depends
from .dependencies.clerk_auth import get_current_user
load_dotenv()


# This is a SQLAlchemy command that creates all the database tables defined in your SQLAlchemy models if they do not already exist.
# models.Base.metadata.create_all(bind=engine) 

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/debug-auth")
def debug_auth(user=Depends(get_current_user)):
    return {"user_id": user.id, "email": user.email}


app.include_router(user.router)
app.include_router(chats.router)
app.include_router(messages.router)
app.include_router(images.router)