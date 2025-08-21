from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from ..query_processor.rag_query_processor import RagQueryProcessor
from ..query_processor.fine_tune_query_processor import FineTuneQueryProcessor
from ..rag.memory import MemoryBank
from ..rag.index import RagIndex
from typing import Dict, List
from ..service.chart_boat_service import process_query

index_dir = Path("data/index")
memory_path = Path("data/memory/memory_bank.csv")
model_size = "small"
model_path = "fine_tune_models/"
mode = "rag"
    
memory_bank = MemoryBank(memory_path)
index = None

index = RagIndex.load(index_dir)
query_processor_rag = RagQueryProcessor(model_size=model_size)
query_processor_fine_tune = FineTuneQueryProcessor(model_path=model_path)

def finetuned_answer(query: str)  -> Dict[str, str]:
     return  process_query(query, query_processor_fine_tune, memory_bank)

def rag_answer(query: str)  -> Dict[str, str]:
    return process_query(query, query_processor_rag, memory_bank, index)


# === FastAPI App ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 template setup
templates = Jinja2Templates(directory="src/web/templates")

class QueryRequest(BaseModel):
    query: str
    mode: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(req: QueryRequest):
    query = req.query.strip()
    mode = req.mode.lower()

    if mode == "rag":
        model_response = rag_answer(query)
    elif mode == "finetune":
        model_response = finetuned_answer(query)
    else:
        response = {"error": "Invalid mode selected. Choose either rag or finetune."}


    if not model_response.get("success", False):
        response = {"error": response.get("error", "Unknown error occurred")}
    else:
        response = {
            "answer": model_response.get("answer", "No answer provided"),
            "success": True
        }

    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
