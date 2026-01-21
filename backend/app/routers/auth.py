from fastapi import APIRouter,Depends,status,HTTPException,Response
from sqlalchemy.orm import Session
from .. import models,Schemas
from ..dependencies import utils,oauth2
from ..database import get_db
from typing import List
from fastapi.security.oauth2 import OAuth2PasswordRequestForm

router = APIRouter(
    prefix="/api/login",
    tags=["Authentication"] 
)

@router.post('/',response_model=Schemas.Token)
# user_credential: Schemas.UserLogin -> user_credential:OAuth2PasswordRequestForm = Depends()
async def login(user_credential: OAuth2PasswordRequestForm = Depends(),db: Session=Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == user_credential.username).first() # user_credential.email -> user_credential.username
    if not user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,detail="Invalid email")
    else:
        if not utils.verify(user_credential.password,user.password):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,detail="Invalid password")
        else:
            access_token = oauth2.create_access_token(data={"user_id": user.id,"email":user.email})
            return {"access_token": access_token,"token_type": "bearer"}