import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    sections = [
        "You are classifying podcast episodes by their format.",
        "Classify the episode into exactly one of these four labels:",
        "",
        "- interview: a conversation between a host and one or more guests",
        "- solo: a single host speaking from memory, experience, or opinion — no guests, no assembled external sources",
        "- panel: multiple guests with roughly equal speaking time, often debating or discussing a topic together",
        "- narrative: a story assembled from external sources — interviews, archival audio, reporting — with a clear narrative arc",
        "",
        "Use only the information in the descriptions below.",
        "If the episode is ambiguous, choose the single best label and mention the uncertainty briefly in the reasoning.",
        "Return exactly two lines and nothing else:",
        "Label: <one valid label>",
        "Reasoning: <one brief explanation>",
        "Do not use JSON, bullet points, or code fences.",
    ]

    if labeled_examples:
        sections.extend(["", "Labeled examples:"])
        for example in labeled_examples:
            title = str(example.get("title", "")).strip() or "Untitled"
            example_description = str(example.get("description", "")).strip() or "No description provided."
            label = str(example.get("label", "")).strip()
            sections.extend([
                "---",
                f"Title: {title}",
                f"Description: {example_description}",
                f"Label: {label}",
            ])
    else:
        sections.extend([
            "",
            "There are no labeled examples available, so classify the episode zero-shot using the definitions above.",
        ])

    target_description = str(description).strip() or "No description provided."
    sections.extend([
        "",
        "Episode to classify:",
        f"Description: {target_description}",
        "Label: ?",
        "",
        "Classify the episode above. Return your answer in this exact format:",
        "Label: <one valid label>",
        "Reasoning: <one brief explanation>",
    ])

    return "\n".join(sections)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )

        response_text = (response.choices[0].message.content or "").strip()
        #temporary debugging output to see raw LLM response
        print(response_text)
        lines = [line.strip() for line in response_text.splitlines() if line.strip()]

        label = None
        reasoning = None

        for index, line in enumerate(lines):
            lower_line = line.lower()

            if lower_line.startswith("label:"):
                label = line.split(":", 1)[1].strip().lower()
            elif lower_line.startswith("reasoning:"):
                reasoning_text = line.split(":", 1)[1].strip()
                continuation_lines = []
                for extra_line in lines[index + 1:]:
                    if ":" in extra_line and extra_line.split(":", 1)[0].strip().lower() in {"label", "reasoning"}:
                        break
                    continuation_lines.append(extra_line)

                reasoning_parts = [part for part in [reasoning_text, *continuation_lines] if part]
                reasoning = " ".join(reasoning_parts).strip() if reasoning_parts else ""

        if label is None or reasoning is None:
            return {
                "label": "unknown",
                "reasoning": "Unparseable or invalid model response.",
            }

        if label not in VALID_LABELS:
            return {
                "label": "unknown",
                "reasoning": reasoning or "Unparseable or invalid model response.",
            }

        return {
            "label": label,
            "reasoning": reasoning or "No reasoning provided.",
        }
    except Exception as exc:
        print(f"[LLM ERROR] {exc}")
        return {
            "label": "unknown",
            "reasoning": f"LLM classification failed: {str(exc).strip() or 'unknown error'}",
        }
