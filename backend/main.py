from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .llm_clients import call_gpt, call_claude, call_gemini
from .voting import majority_vote, weighted_vote, meta_vote

app = FastAPI()

# Allow frontend calls (you can restrict origins later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev this is fine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"


@app.get("/")
async def serve_index():
    """
    Serve the main HTML page.
    """
    return FileResponse(INDEX_FILE)


from io import BytesIO, StringIO

@app.post("/api/label")
async def label_requirements(
    file: UploadFile = File(...),
    voting_method: str = Form("majority"),
    weight_gpt: Optional[float] = Form(None),
    weight_claude: Optional[float] = Form(None),
    weight_gemini: Optional[float] = Form(None),
    judge_model: Optional[str] = Form(None),
):
    """
    Accept a CSV or XLSX/XLS file with columns: issue_id, title, description
    Run 3 LLMs + voting (majority, weighted, or meta) and return JSON.
    
    Args:
        file: CSV or Excel file with requirements
        voting_method: "majority", "weighted", or "meta"
        weight_gpt: Weight for GPT (required if voting_method is "weighted")
        weight_claude: Weight for Claude (required if voting_method is "weighted")
        weight_gemini: Weight for Gemini (required if voting_method is "weighted")
        judge_model: Which LLM to use as judge - "gpt", "claude", or "gemini" (required if voting_method is "meta")
    """
    filename = (file.filename or "").lower()

    try:
        # Read the raw bytes from the uploaded file once
        contents: bytes = await file.read()

        if filename.endswith(".csv"):
            # Decode bytes -> text, then wrap in StringIO for pandas
            text = contents.decode("utf-8")
            df = pd.read_csv(StringIO(text))

        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            # Wrap raw bytes in BytesIO for pandas.read_excel
            df = pd.read_excel(BytesIO(contents))

        else:
            return JSONResponse(
                status_code=400,
                content={"error": "File must be .csv, .xlsx, or .xls"},
            )

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to read file: {str(e)}"},
        )

    # Validate expected columns
    if "issue_id" not in df.columns:
        return JSONResponse(
            status_code=400,
            content={"error": "File must contain an 'issue_id' column."},
        )

    # Validate voting method
    if voting_method not in ["majority", "weighted", "meta"]:
        return JSONResponse(
            status_code=400,
            content={"error": "voting_method must be 'majority', 'weighted', or 'meta'"},
        )
    
    # Validate judge model if meta-voting is selected
    if voting_method == "meta":
        if judge_model not in ["gpt", "claude", "gemini"]:
            return JSONResponse(
                status_code=400,
                content={"error": "judge_model must be 'gpt', 'claude', or 'gemini' for meta-voting"},
            )
    
    # Validate weights if weighted voting is selected
    if voting_method == "weighted":
        if weight_gpt is None or weight_claude is None or weight_gemini is None:
            return JSONResponse(
                status_code=400,
                content={"error": "All weights (weight_gpt, weight_claude, weight_gemini) are required for weighted voting"},
            )
        # Normalize weights (they should sum to 1, but we'll normalize them)
        total_weight = weight_gpt + weight_claude + weight_gemini
        if total_weight <= 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Weights must sum to a positive number"},
            )
        # Normalize to sum to 1
        weight_gpt = weight_gpt / total_weight
        weight_claude = weight_claude / total_weight
        weight_gemini = weight_gemini / total_weight
        weights = [weight_gpt, weight_claude, weight_gemini]
    else:
        weights = None

    # Convert to list of dicts for LLM calls
    requirements: List[Dict] = df.to_dict(orient="records")

    # Call each model
    print(f"DEBUG MAIN: Calling LLMs for {len(requirements)} requirements")
    gpt_labels = call_gpt(requirements)
    claude_labels = call_claude(requirements)
    gemini_labels = call_gemini(requirements)
    
    print(f"DEBUG MAIN: GPT labels returned: {gpt_labels}")
    print(f"DEBUG MAIN: Claude labels returned: {claude_labels}")
    print(f"DEBUG MAIN: Gemini labels returned: {gemini_labels}")

    results = []
    for r in requirements:
        issue_id = r.get("issue_id")
        title = r.get("title", "")
        description = r.get("description", "")

        lg = gpt_labels.get(issue_id)
        lc = claude_labels.get(issue_id)
        lge = gemini_labels.get(issue_id)
        
        print(f"DEBUG MAIN [{issue_id}]: GPT label = {repr(lg)}, Claude label = {repr(lc)}, Gemini label = {repr(lge)}")

        # Apply selected voting method
        if voting_method == "weighted":
            final = weighted_vote([lg, lc, lge], weights)
        elif voting_method == "meta":
            final = meta_vote(r, lg, lc, lge, judge_model)
        else:
            final = majority_vote([lg, lc, lge])

        results.append({
            "issue_id": issue_id,
            "title": title,
            "description": description,
            "label_gpt": lg,
            "label_claude": lc,
            "label_gemini": lge,
            "final_label": final,
        })

    return {"results": results}
