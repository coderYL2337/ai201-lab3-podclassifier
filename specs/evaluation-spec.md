# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
Accuracy = (# of positions i where predictions[i] == ground_truth[i]) / (total # of episodes).
"Correct" means the predicted label exactly equals the ground-truth label for the same episode.
Divide by len(ground_truth) (same as len(predictions) when inputs are aligned).
```

---

**Step-by-step logic:**

```
1. If there are no episodes, return 0.0 to avoid dividing by zero.
2. Loop through predictions and ground_truth in parallel and count exact matches.
3. Return correct_count / total_count.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. There are no evaluated examples, so accuracy is defined as 0.0 here,
and this avoids a division-by-zero error.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Compare each position:
1) interview == interview   -> correct
2) solo == solo             -> correct
3) panel != solo            -> incorrect
4) interview != narrative   -> incorrect

correct_count = 2
total_count = 4
compute_accuracy() = 2 / 4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
For a label L, an episode is counted as "correct" for that class if:
1) its ground-truth label is L, and
2) its predicted label is also L.

Example for "interview": count an episode only when truth == "interview"
and prediction == "interview".
```

---

**What does "total" mean for a given class?**

```
For a label L, "total" is the number of episodes whose ground-truth label is L.
It is not the total number of predictions across all classes.
```

---

**Step-by-step logic:**

```
1. Initialize a results dict for every label in VALID_LABELS with
   {"correct": 0, "total": 0, "accuracy": 0.0}.
2. Loop over predictions and ground_truth together.
3. For each pair (predicted, truth):
   - Increment results[truth]["total"] by 1.
   - If predicted == truth, increment results[truth]["correct"] by 1.
4. After the loop, for each label:
   - If total > 0, set accuracy = correct / total.
   - If total == 0, keep accuracy at 0.0.
5. Return the results dict.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy to 0.0, matching the evaluate.py docstring
("accuracy": correct / total (0.0 if total is 0)).
This avoids division by zero and keeps output shape consistent for all labels.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Pair-by-pair:
1) pred=interview, truth=interview  -> interview total+1, correct+1
2) pred=interview, truth=solo       -> solo total+1, correct+0
3) pred=solo,      truth=solo       -> solo total+1, correct+1
4) pred=panel,     truth=panel      -> panel total+1, correct+1
5) pred=panel,     truth=narrative  -> narrative total+1, correct+0

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

Answer:
Per-class accuracy shows where the model is weak, while overall accuracy can hide
that weakness by averaging everything together. For example, a model can score
high overall by doing well on three classes but still fail badly on one class.
Per-class metrics make that imbalance visible and actionable.

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

Answer:
It usually means the prompt/examples are not giving enough clear signals that
separate panel from interview. Possible issues are: too few panel examples,
noisy or inconsistent panel labels, or descriptions that emphasize "conversation"
without highlighting multiple equal speakers. It suggests improving label quality
and adding clearer panel-vs-interview contrast in few-shot examples.

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?

Answer:
With 100 training labels, the classifier would likely improve because the prompt
would contain more diverse patterns and fewer accidental biases from tiny samples.
With 200 test episodes, the measured accuracy would usually be more stable and
trustworthy (lower variance), because each class estimate would be based on more
examples instead of only five.
