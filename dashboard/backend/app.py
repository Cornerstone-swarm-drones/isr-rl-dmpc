"""ISR-RL-DMPC Dashboard – FastAPI backend."""

import os
import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config_api import router as config_router
from .mission_api import router as mission_router
from .training_api import router as training_router
from .data_api import router as data_router

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root
FRONTEND_DIST = ROOT / "dashboard" / "frontend" / "dist"

app = FastAPI(title="ISR-RL-DMPC Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(config_router, prefix="/api/config", tags=["config"])
app.include_router(mission_router, prefix="/api/mission", tags=["mission"])
app.include_router(training_router, prefix="/api/training", tags=["training"])
app.include_router(data_router, prefix="/api/data", tags=["data"])

if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file = FRONTEND_DIST / full_path
        if file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
