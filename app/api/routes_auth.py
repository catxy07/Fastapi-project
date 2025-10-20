from fastapi import APIRouter
from pydantic import BaseModel
from app.core.security import create_token

router = APIRouter()

class AuthiInput(BaseModel):
    username: str
    password: str
    
    
@router.post("/login")
def login(auth_input: AuthiInput):
    if auth_input.username == "admin" and auth_input.password == "admin":
        token = create_token({"sub": auth_input.username})
        return {"access_token": token, "token_type": "bearer"}
    else:
        return {"error": "Invalid credentials"}

    
    