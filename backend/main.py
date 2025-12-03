from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv
import subprocess
import tempfile
import shutil
import os

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

# Directory for temporary cloned repositories
TEMP_REPO_DIR = PROJECT_ROOT / "temp_repos"
TEMP_REPO_DIR.mkdir(exist_ok=True)


@app.get("/")
async def serve_index():
    """
    Serve the main HTML page.
    """
    return FileResponse(INDEX_FILE)


from io import BytesIO, StringIO

def clone_and_extract_repo_context(repo_url: str) -> Optional[str]:
    """
    Clone a git repository and extract relevant context files.
    Returns a string containing relevant code and documentation context.
    """
    if not repo_url:
        return None
    
    # Create a temporary directory for this clone
    temp_dir = tempfile.mkdtemp(dir=TEMP_REPO_DIR)
    
    try:
        # Clone the repository
        print(f"Cloning repository: {repo_url}")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"Error cloning repository: {result.stderr}")
            return None
        
        # Extract relevant files
        context_parts = []
        
        # Read README files
        readme_files = list(Path(temp_dir).rglob("README*"))
        for readme in readme_files[:3]:  # Limit to first 3 README files
            try:
                content = readme.read_text(encoding='utf-8', errors='ignore')
                if len(content) > 5000:  # Truncate very long READMEs
                    content = content[:5000] + "..."
                context_parts.append(f"=== {readme.relative_to(temp_dir)} ===\n{content}\n")
            except Exception as e:
                print(f"Error reading {readme}: {e}")
        
        # Read code files (limit to common file types and reasonable size)
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb', '.php', '.tsx', '.jsx'}
        code_files = []
        for ext in code_extensions:
            code_files.extend(Path(temp_dir).rglob(f"*{ext}"))
        
        # Filter out common directories to ignore
        ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env', 'dist', 'build', '.next'}
        filtered_files = [
            f for f in code_files 
            if not any(ignore_dir in str(f) for ignore_dir in ignore_dirs)
        ]
        
        # Limit to first 20 code files and max 1000 lines per file
        for code_file in filtered_files[:20]:
            try:
                content = code_file.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                if len(lines) > 1000:
                    content = '\n'.join(lines[:1000]) + "\n... (truncated)"
                
                # Only include if file is not too large
                if len(content) < 50000:  # Skip files larger than 50KB
                    context_parts.append(f"=== {code_file.relative_to(temp_dir)} ===\n{content}\n")
            except Exception as e:
                print(f"Error reading {code_file}: {e}")
        
        # Combine all context
        context = "\n\n".join(context_parts)
        
        # Limit total context size (keep it reasonable for LLM)
        if len(context) > 100000:  # ~100KB max
            context = context[:100000] + "\n... (context truncated)"
        
        return context if context else None
        
    except subprocess.TimeoutExpired:
        print(f"Timeout cloning repository: {repo_url}")
        return None
    except Exception as e:
        print(f"Error processing repository: {e}")
        return None
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error cleaning up temp directory: {e}")


@app.post("/api/label")
async def label_requirements(
    file: UploadFile = File(...),
    voting_method: str = Form("majority"),
    weight_gpt: Optional[float] = Form(None),
    weight_claude: Optional[float] = Form(None),
    weight_gemini: Optional[float] = Form(None),
    judge_model: Optional[str] = Form(None),
    repo_url: Optional[str] = Form(None),
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

    # Extract repository context if URL is provided
    repo_context = None
    if repo_url:
        print(f"Extracting context from repository: {repo_url}")
        repo_context = clone_and_extract_repo_context(repo_url)
        if repo_context:
            print(f"Extracted {len(repo_context)} characters of repository context")
        else:
            print("Warning: Failed to extract repository context, proceeding without it")

    # Call each model with repository context
    print(f"DEBUG MAIN: Calling LLMs for {len(requirements)} requirements")
    gpt_labels = call_gpt(requirements, repo_context=repo_context)
    claude_labels = call_claude(requirements, repo_context=repo_context)
    gemini_labels = call_gemini(requirements, repo_context=repo_context)
    
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
            final = meta_vote(r, lg, lc, lge, judge_model, repo_context=repo_context)
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
