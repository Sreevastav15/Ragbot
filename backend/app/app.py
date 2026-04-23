from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.database import Base, engine
from app.routes import upload, answer, chat_history, delete, eval_routes
import os
from dotenv import load_dotenv

load_dotenv()

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(title="RAGBot Pro – Document QA Backend")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB bootstrap ──────────────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(answer.router)
app.include_router(chat_history.router)
app.include_router(delete.router)
app.include_router(eval_routes.router)


@app.get("/health")
def health():
    return {"status": "ok"}