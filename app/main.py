from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api import routes_auth, routes_predict
from app.core.config import settings
from app.middleware.logging_middleware import LoggingMiddleware
from app.core.exceptions import register_exception_handlers

app = FastAPI(title=settings.PROJECT_NAME)

# link middleware
app.add_middleware(LoggingMiddleware)

app.include_router(routes_auth.router, tags=["auth"])
app.include_router(routes_predict.router, tags=["predict"])

# monitoring using Prometheus
Instrumentator().instrument(app).expose(app)

# add exception handlers
register_exception_handlers(app)