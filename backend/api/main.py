from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import rag, health

app = FastAPI(title="RAG Stock API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag.router, prefix="/rag")
app.include_router(health.router, prefix="/health")