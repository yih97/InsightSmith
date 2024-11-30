from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from api.routers import formsGenerator, sam, snapform

def create_app() -> FastAPI:
    app = FastAPI()
    # app.include_router(sam.router)
    # app.include_router(formsGenerator.router)
    app.include_router(snapform.router)

    app.openapi_schema = get_openapi(
        title="VisionForge API",
        version="0.1.0",
        description="Supports Real-time Image Generation API",
        routes=app.routes,
    )

    return app

app = create_app()