import os
from typing import Dict, List, Optional
import openai


Requirement = Dict[str, str]  # e.g., {"issue_id": "...", "title": "...", "description": "..."}


# Optimized concise prompt to reduce token usage
PRIORITIZATION_PROMPT = """Prioritize: {title}{description}

P1=Critical, P2=Medium, P3=Low. Respond with only: P1, P2, or P3."""

# Prompt with repository context
PRIORITIZATION_PROMPT_WITH_CONTEXT = """You are prioritizing requirements for a software project. Below is relevant context from the codebase:

{repo_context}

---

Now prioritize this requirement: {title}{description}

P1=Critical, P2=Medium, P3=Low. Respond with only: P1, P2, or P3."""


def _truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length, preserving words."""
    if not text or len(text) <= max_length:
        return text
    # Truncate at word boundary
    truncated = text[:max_length].rsplit(' ', 1)[0]
    return truncated + "..." if len(text) > max_length else text


def _build_prompt(requirement: Requirement, repo_context: Optional[str] = None) -> str:
    """Build the prompt for a single requirement with optimized token usage."""
    title = requirement.get("title", "").strip()
    description = requirement.get("description", "").strip()
    
    # Truncate description to save tokens (keep title as-is, it's usually short)
    description = _truncate_text(description, max_length=150)
    
    # Use context-aware prompt if repository context is provided
    if repo_context:
        desc_text = f"\n{description}" if description else ""
        return PRIORITIZATION_PROMPT_WITH_CONTEXT.format(
            repo_context=repo_context,
            title=title,
            description=desc_text
        )
    else:
        # Format: if description exists, add it, otherwise just title
        if description:
            return PRIORITIZATION_PROMPT.format(
                title=title,
                description=f"\n{description}"
            )
        else:
            return PRIORITIZATION_PROMPT.format(
                title=title,
                description=""
            )


def _get_openrouter_client():
    """Get an OpenAI-compatible client configured for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("WARNING: OPENROUTER_API_KEY not found in environment variables. Using stub responses.")
        return None
    
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def call_gpt(requirements: List[Requirement], repo_context: Optional[str] = None) -> Dict[str, str]:
    """
    Call GPT model to label each requirement.
    Return a dict: {issue_id: priority_label}
    Uses OpenRouter API (GPT-5 - fast, non-reasoning model).
    """
    client = _get_openrouter_client()
    if not client:
        # Fallback to stub if API key not set
        print("WARNING: Using stub responses for GPT (P2 for all requirements)")
        labels = {}
        for r in requirements:
            issue_id = r.get("issue_id")
            if issue_id:
                labels[issue_id] = "P2"
        return labels

    print(f"INFO: Calling GPT API for {len(requirements)} requirements")
    labels = {}

    for r in requirements:
        issue_id = r.get("issue_id")
        if not issue_id:
            continue

        try:
            prompt = _build_prompt(r, repo_context=repo_context)
            print(f"DEBUG GPT [{issue_id}]: Sending prompt: {prompt[:100]}...")
            
            response = client.chat.completions.create(
                model="openai/gpt-5-nano",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500,  # Minimum required by API, just need P1, P2, or P3
            )
            
            msg = response.choices[0].message
            choice = response.choices[0]
            
            print(f"DEBUG GPT [{issue_id}]: Finish reason: {choice.finish_reason}")
            print(f"DEBUG GPT [{issue_id}]: Message content: {repr(msg.content)}")
            print(f"DEBUG GPT [{issue_id}]: Message content type: {type(msg.content)}")
            print(f"DEBUG GPT [{issue_id}]: Message dir: {[x for x in dir(msg) if not x.startswith('_')]}")
            
            # Check all message attributes
            for attr in dir(msg):
                if not attr.startswith('_') and not callable(getattr(msg, attr, None)):
                    try:
                        val = getattr(msg, attr, None)
                        if val is not None:
                            print(f"DEBUG GPT [{issue_id}]: msg.{attr} = {repr(val)}")
                    except:
                        pass
            
            # Try multiple ways to get content
            label = ""
            if hasattr(msg, 'content') and msg.content:
                label = str(msg.content).strip().upper()
            elif hasattr(msg, 'text') and msg.text:
                label = str(msg.text).strip().upper()
            elif hasattr(choice, 'text') and choice.text:
                label = str(choice.text).strip().upper()
            elif hasattr(choice, 'message') and hasattr(choice.message, 'content') and choice.message.content:
                label = str(choice.message.content).strip().upper()
            
            print(f"DEBUG GPT [{issue_id}]: Extracted label: {repr(label)}")
            
            # Extract P1, P2, or P3 from response
            if label and ("P1" in label or label == "P1"):
                labels[issue_id] = "P1"
            elif label and ("P2" in label or label == "P2"):
                labels[issue_id] = "P2"
            elif label and ("P3" in label or label == "P3"):
                labels[issue_id] = "P3"
            else:
                print(f"WARNING: GPT returned unexpected/empty label '{label}' for {issue_id}, defaulting to P2")
                labels[issue_id] = "P2"  # Default fallback
        except Exception as e:
            # On error, assign default priority and log
            print(f"ERROR: GPT API call failed for {issue_id}: {type(e).__name__}: {str(e)}")
            labels[issue_id] = "P2"

    return labels


def call_claude(requirements: List[Requirement], repo_context: Optional[str] = None) -> Dict[str, str]:
    """
    Call Claude model to label each requirement.
    Return a dict: {issue_id: priority_label}
    Uses OpenRouter API (Claude Sonnet 4.5).
    """
    client = _get_openrouter_client()
    if not client:
        # Fallback to stub if API key not set
        print("WARNING: Using stub responses for Claude (P1 for all requirements)")
        labels = {}
        for r in requirements:
            issue_id = r.get("issue_id")
            if issue_id:
                labels[issue_id] = "P1"
        return labels

    print(f"INFO: Calling Claude API for {len(requirements)} requirements")
    labels = {}

    for r in requirements:
        issue_id = r.get("issue_id")
        if not issue_id:
            continue

        try:
            prompt = _build_prompt(r)
            response = client.chat.completions.create(
                model="anthropic/claude-sonnet-4.5",
                messages=[
                    {"role": "user", "content": prompt}  # Removed redundant system message
                ],
                temperature=0.2,
                max_tokens=500,  # Minimum required by API, need enough for P1/P2/P3
            )
            label = response.choices[0].message.content.strip().upper()
            # Extract P1, P2, or P3 from response
            if "P1" in label:
                labels[issue_id] = "P1"
            elif "P2" in label:
                labels[issue_id] = "P2"
            elif "P3" in label:
                labels[issue_id] = "P3"
            else:
                print(f"WARNING: Claude returned unexpected label '{label}' for {issue_id}, defaulting to P2")
                labels[issue_id] = "P2"  # Default fallback
        except Exception as e:
            # On error, assign default priority and log the error
            print(f"ERROR: Claude API call failed for {issue_id}: {type(e).__name__}: {str(e)}")
            labels[issue_id] = "P2"

    return labels


def call_gemini(requirements: List[Requirement], repo_context: Optional[str] = None) -> Dict[str, str]:
    """
    Call Gemini model to label each requirement.
    Return a dict: {issue_id: priority_label}
    Uses OpenRouter API (Gemini 1.5 Flash - fast, non-reasoning model).
    """
    client = _get_openrouter_client()
    if not client:
        # Fallback to stub if API key not set
        print("WARNING: Using stub responses for Gemini (P3 for all requirements)")
        labels = {}
        for r in requirements:
            issue_id = r.get("issue_id")
            if issue_id:
                labels[issue_id] = "P3"
        return labels

    print(f"INFO: Calling Gemini API for {len(requirements)} requirements")
    labels = {}

    for r in requirements:
        issue_id = r.get("issue_id")
        if not issue_id:
            continue

        try:
            prompt = _build_prompt(r, repo_context=repo_context)
            response = client.chat.completions.create(
                model="google/gemini-2.5-pro",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500,  # Minimum required by API, just need P1, P2, or P3
            )
            
            msg = response.choices[0].message
            label = msg.content.strip().upper() if msg.content else ""
            print(f"DEBUG GEMINI [{issue_id}]: Response label: {repr(label)}")
            
            # Extract P1, P2, or P3 from response
            if label and ("P1" in label or label == "P1"):
                labels[issue_id] = "P1"
            elif label and ("P2" in label or label == "P2"):
                labels[issue_id] = "P2"
            elif label and ("P3" in label or label == "P3"):
                labels[issue_id] = "P3"
            else:
                print(f"WARNING: Gemini returned unexpected/empty label '{label}' for {issue_id}, defaulting to P2")
                labels[issue_id] = "P2"  # Default fallback
        except Exception as e:
            # On error, assign default priority and log the error
            print(f"ERROR: Gemini API call failed for {issue_id}: {type(e).__name__}: {str(e)}")
            labels[issue_id] = "P2"

    return labels


# Optimized meta-voting prompt
META_VOTING_PROMPT = """Review requirement prioritization:

Req: {title}{description}
{model1_name}: {model1_label}, {model2_name}: {model2_label}

P1=Critical, P2=Medium, P3=Low. Choose final label: P1, P2, or P3."""

# Meta-voting prompt with repository context
META_VOTING_PROMPT_WITH_CONTEXT = """You are reviewing requirement prioritization for a software project. Below is relevant context from the codebase:

{repo_context}

---

Review this requirement: {title}{description}
{model1_name}: {model1_label}, {model2_name}: {model2_label}

P1=Critical, P2=Medium, P3=Low. Choose final label: P1, P2, or P3."""


def call_judge_llm(
    requirement: Requirement,
    model1_name: str,
    model1_label: str,
    model2_name: str,
    model2_label: str,
    judge_model: str,
    repo_context: Optional[str] = None
) -> str:
    """
    Call a specific LLM as a judge to adjudicate between two other models' labels.
    
    Args:
        requirement: The original requirement dict
        model1_name: Name of first model (e.g., "GPT")
        model1_label: Label from first model
        model2_name: Name of second model (e.g., "Claude")
        model2_label: Label from second model
        judge_model: Which model to use as judge ("gpt", "claude", or "gemini")
    
    Returns:
        Final priority label (P1, P2, or P3)
    """
    client = _get_openrouter_client()
    if not client:
        # Fallback: use majority vote if API key not set
        from .voting import majority_vote
        return majority_vote([model1_label, model2_label]) or "P2"
    
    # Map judge_model to actual model identifier (using fast, non-reasoning models)
    model_map = {
        "gpt": "openai/gpt-5-nano",
        "claude": "anthropic/claude-4.5-sonnet",
        "gemini": "google/gemini-2.5-pro"
    }
    
    if judge_model.lower() not in model_map:
        # Invalid judge model, fallback to majority
        from .voting import majority_vote
        return majority_vote([model1_label, model2_label]) or "P2"
    
    model_id = model_map[judge_model.lower()]
    
    try:
        title = requirement.get("title", "").strip()
        description = requirement.get("description", "").strip()
        
        # Truncate description for meta-voting too
        description = _truncate_text(description, max_length=150)
        desc_text = f"\n{description}" if description else ""
        
        # Use context-aware prompt if repository context is provided
        if repo_context:
            prompt = META_VOTING_PROMPT_WITH_CONTEXT.format(
                repo_context=repo_context,
                title=title,
                description=desc_text,
                model1_name=model1_name,
                model1_label=model1_label or "No label",
                model2_name=model2_name,
                model2_label=model2_label or "No label"
            )
        else:
            prompt = META_VOTING_PROMPT.format(
                title=title,
                description=desc_text,
                model1_name=model1_name,
                model1_label=model1_label or "No label",
                model2_name=model2_name,
                model2_label=model2_label or "No label"
            )
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500 if "gpt-5-nano" in model_id else 16,  # Just need P1, P2, or P3
        )
        
        msg = response.choices[0].message
        label = msg.content.strip().upper() if msg.content else ""
        
        # Extract P1, P2, or P3 from response
        if label and ("P1" in label or label == "P1"):
            return "P1"
        elif label and ("P2" in label or label == "P2"):
            return "P2"
        elif label and ("P3" in label or label == "P3"):
            return "P3"
        else:
            # Fallback to majority vote if judge response is unclear
            from .voting import majority_vote
            return majority_vote([model1_label, model2_label]) or "P2"
            
    except Exception as e:
        # On error, fallback to majority vote
        from .voting import majority_vote
        return majority_vote([model1_label, model2_label]) or "P2"
