# LLM Priority Tool

A web application that uses multiple LLMs (GPT, Claude, Gemini) to prioritize requirements from CSV/Excel files, with optional repository context support.

## Prerequisites

- Python 3.9 or higher
- Git (for cloning repositories when providing repo URLs)
- OpenRouter API key (for LLM access)

## Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd llm-priority-tool
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file in the project root:**
   ```bash
   touch .env
   ```

5. **Add your OpenRouter API key to `.env`:**
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```
   
   You can get an API key from [OpenRouter](https://openrouter.ai/). Make sure you have credits/balance in your account.

## Running the Application

1. **Start the FastAPI server:**
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Or if you're in the project root directory:
   ```bash
   python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:8000
   ```

3. **The application should now be running!**

## Usage

1. **Prepare your requirements file:**
   - Create a CSV or Excel file with columns: `issue_id`, `title`, `description`
   - Example CSV:
     ```csv
     issue_id,title,description
     REQ-001,Add user authentication,Implement login and logout functionality
     REQ-002,Add dark mode,Allow users to switch between light and dark themes
     REQ-003,Add search feature,Enable users to search for content
     ```

2. **Use the web interface:**
   - (Optional) Enter a Git repository URL to provide codebase context to the LLMs
   - Upload your requirements file (CSV, XLSX, or XLS)
   - Select a voting method:
     - **Majority Voting**: Simple majority vote among the 3 LLMs
     - **Weighted Voting**: Assign custom weights to each LLM
     - **Meta-Voting**: Use one LLM as a judge to decide between the other two
   - Click "Run Prioritization"

3. **View results:**
   - The results table will show each requirement with:
     - Individual LLM labels (P1, P2, or P3)
     - Final label based on selected voting method

## Priority Labels

- **P1**: Critical priority
- **P2**: Medium priority  
- **P3**: Low priority

## Features

- ✅ Support for CSV, XLSX, and XLS files
- ✅ Multiple LLM integration (GPT, Claude, Gemini)
- ✅ Three voting methods (Majority, Weighted, Meta-Voting)
- ✅ Optional repository context for better prioritization
- ✅ Automatic repository cloning and context extraction
- ✅ Clean, responsive web interface

## Troubleshooting

- **"OPENROUTER_API_KEY not found"**: Make sure your `.env` file exists and contains the API key
- **Repository cloning fails**: Ensure Git is installed and the repository URL is accessible
- **Port already in use**: Change the port with `--port 8080` (or another available port)
- **Import errors**: Make sure you've activated your virtual environment and installed all dependencies

## Development

The application structure:
- `backend/main.py`: FastAPI application and API endpoints
- `backend/llm_clients.py`: LLM client implementations
- `backend/voting.py`: Voting algorithms
- `frontend/index.html`: Web interface

## Notes

- The application uses shallow clones (depth=1) for faster repository cloning
- Repository context is limited to ~100KB to keep prompts manageable
- Temporary cloned repositories are stored in `temp_repos/` and cleaned up automatically
- The application will continue without repository context if cloning fails

