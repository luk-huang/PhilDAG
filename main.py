from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analyze import router as analyze_router
from routes.ask import router as ask_router
import uvicorn


app = FastAPI(title="PhilDAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/analyze", tags=["analyze"])
app.include_router(ask_router, prefix="/ask", tags=["ask"])


if __name__ == "__main__":
    uvicorn.run("main::app", host = "127.0.0.1", port = 8000, reload = True)
