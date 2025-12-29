# For documentation head over to http://127.0.0.1:8000/redoc or http://127.0.0.1:8000/docs route

from typing import Optional,List
from fastapi import status,HTTPException,Depends,APIRouter
from sqlalchemy.orm import Session
from .. import models,Schemas
from ..database import get_db
from ..dependencies.clerk_auth import get_current_user

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)

@router.get('/me',status_code=status.HTTP_200_OK,response_model = Schemas.UserResponse)
async def get_user(db: Session = Depends(get_db),current_user: object = Depends(get_current_user)):
    user = db.query(models.User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="User not found")
    else:
        return user

@router.patch('/me',status_code=status.HTTP_200_OK,response_model = Schemas.UserResponse)
async def update_user(user: Schemas.UserIn,db: Session = Depends(get_db),current_user: object = Depends(get_current_user)):
    existing_user_query = db.query(models.User).filter(models.User.id == current_user.id)
    if not existing_user_query.first():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="User not found")
    
    existing_user_query.update(user.model_dump(exclude_unset=True),synchronize_session=False)
    db.commit()
    return existing_user_query.first()

# @router.post('/',status_code=status.HTTP_201_CREATED,response_model = Schemas.UserResponse)
# async def create_user(user: Schemas.UserSignup,db: Session = Depends(get_db)):
#     hashed_password = utils.hash_password(user.password)
#     user.password = hashed_password
#     new_user = models.User(**user.model_dump())
#     db.add(new_user)
#     db.commit()
#     db.refresh(new_user)
#     return new_user

# @router.get('/{id}',status_code=status.HTTP_200_OK,response_model = Schemas.UserResponse)
# async def get_user(id: int,db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="User not found")
    else:
        return user