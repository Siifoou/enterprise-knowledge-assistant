import json
from collections import Counter

DATASET = "golden_dataset.json"
CANDIDATES = "golden_candidates.json"

VALID_CATEGORIES = {"FR->FR", "EN->EN", "FR->EN", "EN->FR", "unanswerable"}
REQUIRED = {"id", "question", "expected_answer", "acceptable_sources",
            "language", "category", "answerability"}

with open(DATASET, encoding="utf-8") as f:
    dataset = json.load(f)
with open(CANDIDATES, encoding="utf-8") as f:
    candidates = json.load(f)

# Entries already seen in earlier evaluation runs count as calibration data
for item in dataset:
    item.setdefault("split", "calibration")

existing_ids = {item["id"] for item in dataset}
added, skipped = [], []

for cand in candidates:
    if not cand.get("verified"):
        skipped.append(cand["id"])
        continue
    if cand["id"] in existing_ids:
        print(f"! {cand['id']} is already in the dataset, skipped")
        continue
    missing = REQUIRED - set(cand)
    if missing or cand["category"] not in VALID_CATEGORIES:
        print(f"! {cand['id']} has a problem ({missing or cand['category']}), skipped")
        continue
    # the verification flag and evidence note are working fields, not dataset fields
    dataset.append({k: v for k, v in cand.items() if k not in ("verified", "evidence")})
    added.append(cand["id"])

with open(DATASET, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("Merged:", added or "nothing")
print("Not merged (verified is still false):", skipped or "none")
print("\nDataset now has", len(dataset), "entries")
print("By category:", dict(Counter(i["category"] for i in dataset)))
print("By split:   ", dict(Counter(i["split"] for i in dataset)))
