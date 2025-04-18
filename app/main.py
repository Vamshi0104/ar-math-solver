from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes import equation

app = FastAPI(title="AR Math Solver")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")

app.include_router(equation.router)


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ar-markerless", response_class=HTMLResponse)
async def arpage(request: Request):
    return templates.TemplateResponse("ar-markerless.html", {"request": request})


@app.get("/ar-marker", response_class=HTMLResponse)
async def arpage(request: Request):
    return templates.TemplateResponse("ar-marker.html", {"request": request})
