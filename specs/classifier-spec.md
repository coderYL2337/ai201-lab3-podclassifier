# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
Ask for exactly two lines in this format:

Label: <one valid label>
Reasoning: <one brief explanation>

Choice: use the tagged two-line format above.

Tradeoffs:
- "Label: X / Reasoning: Y" is easier to parse than free-form prose because
  each field has a fixed prefix. It is also less brittle than JSON because we
  can parse it with simple string operations even if the model adds minor
  whitespace variation.
- JSON is more structured in theory, but LLMs often wrap JSON in code fences,
  add trailing text, or produce invalid escaping/quotes, which makes parsing
  less reliable unless we add extra cleanup logic.
- "Label on its own line, then explanation" is readable, but harder to parse
  robustly because the first line may contain extra words like "Label" or a
  sentence instead of the raw class name.

Prompt wording should explicitly say:
- Return exactly two lines.
- Use one of these labels exactly as written: interview, solo, panel, narrative.
- Do not use JSON, bullet points, or code fences.
```

---

**Edge cases to handle in the prompt:**

```
If labeled_examples is empty, still provide the task instructions and label
definitions, then classify zero-shot using the same required output format.

If the description is very short or ambiguous, instruct the model to choose the
single best label from the four options using only the provided text and to
briefly note the uncertainty in the Reasoning line.

If the description is empty or whitespace, the prompt can still ask for the
best label, but the implementation should ideally avoid sending such inputs or
expect a low-confidence answer. The prompt should not invite the model to use
outside knowledge.

In all cases, remind the model to return exactly one label and not to explain
the taxonomy beyond the brief reasoning line.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (MODEL_NAME)
  - messages: a list with one dict — {"role": "user", "content": prompt}
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
Strip leading/trailing whitespace from the response text, then split into
lines.

Look for a line that starts with "Label:" and a line that starts with
"Reasoning:". Extract the text after the first colon in each line and strip
whitespace again.

Lowercase the label before validation so values like "Interview" become
"interview".

If the model returns extra lines, ignore them unless they are needed to recover
the Reasoning field. If either required field is missing, treat the response as
unparseable.
```

---

**Step 4 — Validate the label:**

```
If the parsed label is not one of VALID_LABELS after lowercasing and stripping,
set label to "unknown".

Keep the reasoning if it was parseable, since it may still be useful for
debugging. If reasoning was also missing, return a short fallback message like
"Unparseable or invalid model response."
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the LLM call and parsing logic in try/except.

Possible failures:
- API/network error
- missing response content
- response text that does not contain the required Label/Reasoning lines
- an invalid label outside VALID_LABELS

On any exception, return:

{
  "label": "unknown",
  "reasoning": "LLM classification failed: <brief error summary>"
}

If exposing the raw exception feels too noisy, use a shorter stable message such
as "LLM classification failed due to an API or parsing error." The key point is
that one failed call should not crash the evaluation loop.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: Marine Biologist Dr. Amara Diallo on What Coral Bleaching Actually Looks Like
Raw response text:
Label: interview
Reasoning: The episode features a conversation between a host and Dr. Amara Diallo, with the host asking questions and Dr. Diallo sharing her experiences and expertise, which is characteristic of an interview format.
```

**How did you parse the label out of the response?**

```
I stripped the full response text, split it into non-empty lines, then scanned
for the line that starts with "Label:" (case-insensitive via lowercasing each
line first).

When found, I used split(":", 1) to get everything after the first colon,
then strip() and lower() to normalize it (e.g., "Interview" -> "interview").

I validated the normalized value against VALID_LABELS. If it does not match
exactly, I set label to "unknown".
```

**Did any episodes return `"unknown"`? If so, why?**

```
No for this test run. The model returned a valid label ("interview") in the
expected tagged format.
```

**One thing about the output format that surprised you:**

```
The model followed the exact two-line format on the first try, which made the
parser straightforward and avoided extra cleanup logic.
```
