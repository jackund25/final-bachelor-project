from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.prediction import router as prediction_router
from backend.routes.clinical import router as clinical_router
from backend.routes.logbook import router as logbook_router

app = FastAPI(
    title="Diabetes Clinical Decision Support API",
    version="1.0.0",
    description="Backend API for glucose prediction and clinical recommendation.",
)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
# Development: Next.js biasanya berjalan di localhost:3000.
# Nanti origin production diganti sesuai domain PWA.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://diabetes-glucose-monitor-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)
app.include_router(clinical_router)
app.include_router(logbook_router)

# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "diabetes-cdss-api",
    }


@app.get("/")
def root():
    return {
        "message": "Diabetes Clinical Decision Support API",
        "status": "running",
    }