"""FastAPI service. Run with:  uvicorn review_agent.api.main:app --host 0.0.0.0 --port 8000"""
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.language_models import BaseChatModel
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError
from sqlalchemy.engine import Engine
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..agent import AgentRunner, AgentUnavailable, build_chat_model
from ..config import Settings, get_settings
from ..ingestion.importer import import_products, import_reviews, load_directory
from ..observability import Metrics, configure_logging, get_logger, request_id_var, session_id_var, setup_tracing
from ..retrieval.bm25 import BM25Index
from ..storage.db import make_engine, make_session_factory
from ..storage.repository import Repository
from .schemas import ChatRequest, ChatResponse, FeedbackRequest, ImportRequest, SessionView
from .security import RateLimiter, authenticate, authenticate_admin

log = get_logger("review_agent.api")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_CHAT_PAGE = (Path(__file__).parent / "static" / "chat.html").read_text(encoding="utf-8")


@dataclass
class Services:
    settings: Settings
    engine: Engine
    repo: Repository
    index: BM25Index
    metrics: Metrics
    runner: AgentRunner
    limiter: RateLimiter

    def rebuild_index(self) -> None:
        self.index.build(self.repo.all_reviews())
        self.metrics.index_size.set(self.index.size)


def _error(code: str, message: str, http_status: int, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "request_id": request_id_var.get()}},
                        status_code=http_status, headers=headers)


def create_app(settings: Settings | None = None, model: BaseChatModel | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        setup_tracing(settings.service_name, settings.env, settings.otel_exporter, settings.otel_endpoint)
        engine = make_engine(settings.database_url)
        repo = Repository(make_session_factory(engine))
        if settings.seed_sample_data and repo.review_count() == 0:
            report = load_directory(repo, Path(settings.sample_data_dir))
            log.info("sample_data_loaded", extra={"accepted": report.accepted})
        metrics = Metrics()
        index = BM25Index()
        runner = AgentRunner(settings, repo, index, model or build_chat_model(settings), metrics)
        services = Services(settings, engine, repo, index, metrics, runner,
                            RateLimiter(settings.rate_limit_per_minute))
        services.rebuild_index()
        app.state.services = services
        log.info("service_started", extra={"env": settings.env, "provider": settings.llm_provider,
                                           "model": settings.llm_model, "reviews_indexed": index.size})
        yield
        engine.dispose()

    is_prod = settings.env == "prod"
    app = FastAPI(title="Review Insights Agent", version=__version__, lifespan=lifespan,
                  docs_url=None if is_prod else "/docs", redoc_url=None,
                  openapi_url=None if is_prod else "/openapi.json")
    if settings.cors_list:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_methods=["GET", "POST", "DELETE"],
                           allow_headers=["content-type", "x-api-key", "x-request-id"])

    @app.middleware("http")
    async def observe(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        started = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                log.exception("unhandled_error")
                response = _error("internal_error", "Something went wrong. Please retry.", 500)
            route = request.scope.get("route")
            path = route.path if route else "unmatched"
            elapsed = time.perf_counter() - started
            services = getattr(request.app.state, "services", None)
            if services:
                services.metrics.http_requests.labels(request.method, path, str(response.status_code)).inc()
                services.metrics.http_latency.labels(path).observe(elapsed)
            response.headers["X-Request-ID"] = rid
            response.headers["X-Content-Type-Options"] = "nosniff"
            if path not in ("/healthz", "/metrics"):
                log.info("http_request", extra={"method": request.method, "route": path,
                                                "status": response.status_code, "ms": round(elapsed * 1000, 1)})
            return response
        finally:
            request_id_var.reset(token)

    @app.exception_handler(StarletteHTTPException)  # also covers 404/405 raised by the router
    async def http_error(_request: Request, exc: StarletteHTTPException):
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found", 429: "rate_limited"}.get(
            exc.status_code, "bad_request")
        return _error(code, str(exc.detail), exc.status_code, getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        detail = "; ".join(f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in exc.errors())
        return _error("invalid_request", detail, 422)

    @app.exception_handler(AgentUnavailable)
    async def model_down(_request: Request, _exc: AgentUnavailable):
        return _error("model_unavailable", "The AI model is temporarily unavailable. Please retry shortly.", 503,
                      {"Retry-After": "10"})

    def services(request: Request) -> Services:
        return request.app.state.services

    def rate_limited_client(request: Request, client_id: str = Depends(authenticate)) -> str:
        wait = request.app.state.services.limiter.check(client_id)
        if wait:
            raise HTTPException(429, "Too many requests.", headers={"Retry-After": str(int(wait) + 1)})
        return client_id

    # ---- chat page ----------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def chat_page():
        return HTMLResponse(_CHAT_PAGE, headers={"Cache-Control": "no-cache"})

    # ---- public probes ----------------------------------------------------------------
    @app.get("/healthz", tags=["ops"])
    def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    def readyz(svc: Services = Depends(services)):
        try:
            db_ok = svc.repo.ping()
        except Exception:
            db_ok = False
        body = {"database": db_ok, "reviews_indexed": svc.index.size}
        return JSONResponse(body, status_code=200 if db_ok else 503)

    @app.get("/metrics", tags=["ops"])
    def metrics(svc: Services = Depends(services)):
        return Response(generate_latest(svc.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # ---- chat ---------------------------------------------------------------------------
    @app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
    def chat(body: ChatRequest, svc: Services = Depends(services), client_id: str = Depends(rate_limited_client)):
        session_id = body.session_id or uuid.uuid4().hex
        token = session_id_var.set(session_id)
        try:
            r = svc.runner.run(client_id=client_id, session_id=session_id, message=body.message,
                               request_id=request_id_var.get())
        except PermissionError as exc:
            raise HTTPException(404, "Session not found.") from exc
        finally:
            session_id_var.reset(token)
        return ChatResponse(request_id=r.request_id, session_id=r.session_id, status=r.status, answer=r.answer,
                            citations=r.citations, latency_ms=r.latency_ms, guardrail_flags=r.flags,
                            usage={"input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                                   "cost_usd": r.cost_usd})

    @app.get("/v1/sessions/{session_id}", response_model=SessionView, tags=["chat"])
    def get_session(session_id: str, svc: Services = Depends(services), client_id: str = Depends(authenticate)):
        s = svc.repo.get_session(session_id, client_id)
        if s is None:
            raise HTTPException(404, "Session not found.")
        return SessionView(session_id=s.id, summary=s.summary, profile=s.profile or {},
                           messages=[{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
                                     for m in svc.repo.messages(session_id)])

    @app.delete("/v1/sessions/{session_id}", status_code=204, tags=["chat"])
    def delete_session(session_id: str, svc: Services = Depends(services), client_id: str = Depends(authenticate)):
        if not svc.repo.delete_session(session_id, client_id):  # right to erasure: history and profile go
            raise HTTPException(404, "Session not found.")
        return Response(status_code=204)

    @app.post("/v1/feedback", status_code=202, tags=["chat"])
    def feedback(body: FeedbackRequest, svc: Services = Depends(services), client_id: str = Depends(authenticate)):
        session_id = svc.repo.run_session(body.request_id)
        if session_id is None or svc.repo.get_session(session_id, client_id) is None:
            raise HTTPException(404, "Request not found.")
        svc.repo.save_feedback(body.request_id, body.score, body.comment)
        svc.metrics.feedback.labels("up" if body.score > 0 else "down").inc()
        return {"status": "recorded"}

    # ---- admin --------------------------------------------------------------------------
    @app.post("/v1/admin/reviews/import", tags=["admin"], dependencies=[Depends(authenticate_admin)])
    def import_data(body: ImportRequest, svc: Services = Depends(services)):
        try:
            products = import_products(svc.repo, body.products) if body.products else 0
        except ValidationError as exc:
            raise HTTPException(422, f"Invalid product: {exc.errors()[0]['msg']}") from exc
        report = import_reviews(svc.repo, body.reviews)
        if report.accepted:
            svc.rebuild_index()
        return {"products_upserted": products, **report.model_dump(), "reviews_indexed": svc.index.size}

    @app.get("/v1/admin/runs", tags=["admin"], dependencies=[Depends(authenticate_admin)])
    def runs(svc: Services = Depends(services), limit: int = Query(50, ge=1, le=500), flagged_only: bool = False):
        return {"runs": svc.repo.recent_runs(limit, flagged_only)}

    return app


_app: FastAPI | None = None


def __getattr__(name: str):  # `uvicorn review_agent.api.main:app` builds the app on first access
    global _app
    if name == "app":
        _app = _app or create_app()
        return _app
    raise AttributeError(name)
