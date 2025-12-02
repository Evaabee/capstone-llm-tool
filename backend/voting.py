from typing import List, Dict, Optional

# Map labels to numeric order for tie-breaking
LABEL_ORDER = {
    "P1": 1,  # highest
    "P2": 2,
    "P3": 3,  # lowest
}


def normalize_label(label: str) -> str:
    """
    Normalize a label string to one of: 'P1', 'P2', 'P3'.
    You can expand this later if models return variants.
    """
    if label is None:
        return None

    label = label.strip().upper()

    # Simple normalizations; customize if needed
    if label in ("P1", "PRIORITY1", "HIGH"):
        return "P1"
    if label in ("P2", "PRIORITY2", "MEDIUM"):
        return "P2"
    if label in ("P3", "PRIORITY3", "LOW"):
        return "P3"

    # Fallback: if unknown, just return as-is so you can debug
    return label


def majority_vote(labels: List[str]) -> str:
    """
    Given a list of labels like ['P1', 'P2', 'P1'], return the majority.
    - If any label has count >= 2, use that.
    - If all three disagree (P1, P2, P3), fallback to the median priority.
    """
    # Clean and filter out None
    labels = [normalize_label(lbl) for lbl in labels if lbl is not None]

    if not labels:
        return None

    # Check for simple majority
    unique_labels = set(labels)
    for lbl in unique_labels:
        if labels.count(lbl) >= 2:
            return lbl

    # No majority: pick the "median" label by numeric order
    def label_key(lbl: str) -> int:
        return LABEL_ORDER.get(lbl, 999)  # unknown labels go to the end

    sorted_labels = sorted(labels, key=label_key)

    # Median element
    return sorted_labels[len(sorted_labels) // 2]


def weighted_vote(
    labels: List[str],
    weights: List[float],
    label_order: Optional[Dict[str, int]] = None
) -> str:
    """
    Given a list of labels and corresponding weights, return the weighted majority.
    
    Args:
        labels: List of labels from each LLM (e.g., ['P1', 'P2', 'P1'])
        weights: List of weights for each LLM (e.g., [0.5, 0.3, 0.2])
        label_order: Optional dict mapping labels to numeric values for tie-breaking
    
    Returns:
        The label with the highest weighted score
    """
    if label_order is None:
        label_order = LABEL_ORDER
    
    # Clean and filter out None
    valid_pairs = [(normalize_label(lbl), w) for lbl, w in zip(labels, weights) if lbl is not None]
    
    if not valid_pairs:
        return None
    
    # Calculate weighted scores for each priority level
    scores = {"P1": 0.0, "P2": 0.0, "P3": 0.0}
    
    for label, weight in valid_pairs:
        if label in scores:
            scores[label] += weight
    
    # Find the label with the highest score
    max_score = max(scores.values())
    winners = [label for label, score in scores.items() if score == max_score]
    
    # If there's a tie, use the label order to break it (prefer higher priority)
    if len(winners) > 1:
        winners.sort(key=lambda lbl: label_order.get(lbl, 999))
    
    return winners[0] if winners else None


def meta_vote(
    requirement: Dict[str, str],
    gpt_label: str,
    claude_label: str,
    gemini_label: str,
    judge_model: str
) -> str:
    """
    Use one LLM as a judge to adjudicate between the other two models' labels.
    
    Args:
        requirement: The original requirement dict
        gpt_label: Label from GPT
        claude_label: Label from Claude
        gemini_label: Label from Gemini
        judge_model: Which model to use as judge ("gpt", "claude", or "gemini")
    
    Returns:
        Final priority label (P1, P2, or P3)
    """
    from .llm_clients import call_judge_llm
    
    # Determine which two models are being judged
    if judge_model.lower() == "gpt":
        # GPT is judge, Claude and Gemini are being judged
        return call_judge_llm(
            requirement=requirement,
            model1_name="Claude",
            model1_label=claude_label,
            model2_name="Gemini",
            model2_label=gemini_label,
            judge_model="gpt"
        )
    elif judge_model.lower() == "claude":
        # Claude is judge, GPT and Gemini are being judged
        return call_judge_llm(
            requirement=requirement,
            model1_name="GPT",
            model1_label=gpt_label,
            model2_name="Gemini",
            model2_label=gemini_label,
            judge_model="claude"
        )
    elif judge_model.lower() == "gemini":
        # Gemini is judge, GPT and Claude are being judged
        return call_judge_llm(
            requirement=requirement,
            model1_name="GPT",
            model1_label=gpt_label,
            model2_name="Claude",
            model2_label=claude_label,
            judge_model="gemini"
        )
    else:
        # Invalid judge model, fallback to majority vote
        return majority_vote([gpt_label, claude_label, gemini_label])
