from fastapi import FastAPI
from dotenv import load_dotenv
from .database import engine, Base

load_dotenv()

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, HR Agentic System"}
