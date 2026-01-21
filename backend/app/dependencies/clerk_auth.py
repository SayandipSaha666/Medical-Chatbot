# from fastapi import Depends, HTTPException, Header, status
# from jose import jwt, JWTError
# import requests
# from sqlalchemy.orm import Session
# from  ..config import settings
# from .. import models
# from ..database import get_db
# from .. import models

# CLERK_JWKS_URL = settings.clerk_jwks_url
# CLERK_ISSUER = settings.clerk_issuer

# jwks = requests.get(CLERK_JWKS_URL).json()

# def get_current_user(
#     authorization: str = Header(...),
#     db: Session = Depends(get_db)
# ):
#     if not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Invalid authorization header")

#     token = authorization.replace("Bearer ", "")

#     try:
#         payload = jwt.decode(
#             token,
#             jwks,
#             algorithms=["RS256"],
#             issuer=CLERK_ISSUER,
#             options={"verify_aud": False}
#         )
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Invalid Clerk token")

#     clerk_user_id = payload["sub"]
#     email = payload.get("email")
#     name = payload.get("name")

#     # 🔁 Sync Clerk user → DB
#     user = db.query(models.User).filter(
#         models.User.clerk_user_id == clerk_user_id
#     ).first()

#     if not user:
#         user = models.User(
#             clerk_user_id=clerk_user_id,
#             email=email,
#             name=name
#         )
#         db.add(user)
#         db.commit()
#         db.refresh(user)

#     return user



# import os
# import jwt
# from jwt import PyJWKClient
# import requests
# from fastapi import Depends, HTTPException, status
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from sqlalchemy.orm import Session
# from ..database import get_db
# from .. import models
# from ..config import settings

# security = HTTPBearer()

# CLERK_AUDIENCE = "fastapi"
# JWKS_URL = settings.clerk_jwks_url
# CLERK_ISSUER = settings.clerk_issuer

# # _jwks = requests.get(JWKS_URL).json()
# jwks_client = PyJWKClient(JWKS_URL)

# def get_current_user(
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     db: Session = Depends(get_db),
# ):
#     token = credentials.credentials
#     print(f"Received auth token")  # Debug: print first 50 chars

#     try:
#         signing_key = jwks_client.get_signing_key_from_jwt(token).key
#         payload = jwt.decode(
#             token,
#             signing_key,
#             algorithms=["RS256"],
#             audience=CLERK_AUDIENCE,
#             issuer=CLERK_ISSUER
#         )
#     except Exception as e:
#         print(f"JWT decode error: {str(e)}")  # Debug
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail=f"Invalid Clerk token: {str(e)}",
#         )

#     clerk_user_id = payload["sub"]
#     email = payload.get("email") or payload.get("primary_email")
#     name = payload.get("name") or payload.get("username")

#     # Create or fetch local user
#     user = db.query(models.User).filter(models.User.clerk_user_id == clerk_user_id).first()
#     if not user:
#         user = models.User(
#             clerk_user_id=clerk_user_id,
#             email=email,
#             name=name
#         )
#         db.add(user)
#         db.commit()
#         db.refresh(user)

#     return user
