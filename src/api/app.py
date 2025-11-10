"""
FastAPI application for document analysis
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from .middleware import RequestIdMiddleware

app = FastAPI(
    title="Document Analysis API",
    description="API pour l'analyse de documents avec pipeline ML",
    version="0.1.0"
)

# Request ID middleware (doit être ajouté en premier pour capturer toutes les requêtes)
app.add_middleware(RequestIdMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Document Analysis API"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    from src.utils.config import get_config
    
    config = get_config()
    uvicorn.run(
        app,
        host=config.get("API_HOST", "0.0.0.0"),
        port=int(config.get("API_PORT", 8000)),
        reload=config.get("API_DEBUG", False)
    )

