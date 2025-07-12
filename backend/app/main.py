from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from .api.routes import agent

app = FastAPI(
    title="Research Assistant API",
    description="AI-powered research assistant backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router)


@app.get("/")
def read_root():
    return {"message": "Research Assistant API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
