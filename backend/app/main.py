from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from passlib.context import CryptContext
from dotenv import load_dotenv
from .database import SessionLocal, get_db
from .models import User, Employee
from sqlalchemy.orm import Session
from sqlalchemy.orm import Session

load_dotenv()

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

app = FastAPI()

class UserSignup(BaseModel):
    username: str
    password: str
    first_name: str
    last_name: str
    email: str

def get_password_hash(password):
    return pwd_context.hash(password)

@app.get("/")
def read_root():
    return {"message": "Hello, HR Agentic System"}

@app.post("/signup")
def signup(user: UserSignup, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_employee = db.query(Employee).filter(Employee.email == user.email).first()
    if db_employee:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password, role_id=1)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    new_employee = Employee(user_id=new_user.id, first_name=user.first_name, last_name=user.last_name, email=user.email)
    db.add(new_employee)
    db.commit()
    
    return {"message": "User created successfully", "user_id": new_user.id}


