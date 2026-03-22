from fastapi import FastAPI
from api.routes import rag, health

app = FastAPI(title="RAG Stock API")

app.include_router(rag.router, prefix="/rag")
app.include_router(health.router, prefix="/health")