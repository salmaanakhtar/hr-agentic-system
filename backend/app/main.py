from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from dotenv import load_dotenv
from .database import SessionLocal, get_db
from .models import User, Employee, Manager, HR, Role
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
from sqlalchemy.orm import Session

load_dotenv()

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class LoginCredentials(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    try:
        roles = ["employee", "manager", "hr", "admin"]
        for role_name in roles:
            role = db.query(Role).filter(Role.name == role_name).first()
            if not role:
                new_role = Role(name=role_name)
                db.add(new_role)
                print(f"Seeded role: {role_name}")
        db.commit()
        print("Role seeding completed")
    finally:
        db.close()

class UserSignup(BaseModel):
    username: str
    email: str
    password: str
    role: str
    first_name: str
    last_name: str

def get_password_hash(password):
    return pwd_context.hash(password)

@app.get("/")
def read_root():
    return {"message": "Hello, HR Agentic System"}

@app.post("/signup")
def signup(user: UserSignup, db: Session = Depends(get_db)):
    print(f"Signup request: {user}")
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    db_employee = db.query(Employee).filter(Employee.email == user.email).first()
    db_manager = db.query(Manager).filter(Manager.email == user.email).first()
    db_hr = db.query(HR).filter(HR.email == user.email).first()
    
    if db_employee or db_manager or db_hr:
        raise HTTPException(status_code=400, detail="Email already registered")

    role = db.query(Role).filter(Role.name == user.role).first()
    if not role:
        print(f"Role not found: {user.role}")
        raise HTTPException(status_code=400, detail="Invalid role")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password, role_id=role.id)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    

    if user.role == "employee":
        new_record = Employee(user_id=new_user.id, first_name=user.first_name, last_name=user.last_name, email=user.email)
    elif user.role == "manager":
        new_record = Manager(user_id=new_user.id, first_name=user.first_name, last_name=user.last_name, email=user.email)
    elif user.role == "hr":
        new_record = HR(user_id=new_user.id, first_name=user.first_name, last_name=user.last_name, email=user.email)
    else:
        new_record = Employee(user_id=new_user.id, first_name=user.first_name, last_name=user.last_name, email=user.email)
    
    db.add(new_record)
    db.commit()
    
    return {"message": "User created successfully", "user_id": new_user.id}

@app.post("/login", response_model=Token)
async def login(credentials: LoginCredentials, db: Session = Depends(get_db)):
    user = authenticate_user(db, credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Incorrect username or password",
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    role_name = db.query(Role).filter(Role.id == current_user.role_id).first().name

    user_data = None
    if role_name == "employee":
        user_data = db.query(Employee).filter(Employee.user_id == current_user.id).first()
    elif role_name == "manager":
        user_data = db.query(Manager).filter(Manager.user_id == current_user.id).first()
    elif role_name == "hr":
        user_data = db.query(HR).filter(HR.user_id == current_user.id).first()
    else:

        user_data = db.query(Employee).filter(Employee.user_id == current_user.id).first()
    
    if not user_data:
        raise HTTPException(status_code=404, detail="User data not found")
    
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=user_data.email,
        role=role_name
    )


