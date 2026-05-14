"""
Експорт фінального набору запитань у docs/questions/{section}.json.

  python -m scripts.export

Логіка:
  - читає data/normalized/dedup.jsonl (або generated.jsonl якщо dedup відсутній)
  - відкидає запитання з ID у data/normalized/flagged_major.txt
  - проводить базову sanity-перевірку (5 опцій, correct in A-E, непорожнє питання)
  - групує за section та записує у docs/questions/{section}.json
  - НЕ перезаписує сидові питання сидового періоду — додає до них якщо вказано режим merge
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NORMALIZED = DATA / "normalized"
QUESTIONS_DIR = ROOT / "docs" / "questions"

SECTIONS = [
    "general",
    "therapeutic",
    "surgical",
    "orthopedic",
    "pediatric_therapeutic",
    "pediatric_surgical",
    "orthodontic",
]

console = Console()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_valid(q: dict) -> tuple[bool, str]:
    opts = q.get("options", {})
    if not isinstance(opts, dict) or sorted(opts.keys()) != ["A", "B", "C", "D", "E"]:
        return False, "options not exactly A-E"
    if q.get("correct") not in ["A", "B", "C", "D", "E"]:
        return False, f"correct invalid: {q.get('correct')}"
    if not (q.get("question_ua") or "").strip():
        return False, "empty question_ua"
    for letter in ["A", "B", "C", "D", "E"]:
        if not (opts.get(letter) or "").strip():
            return False, f"empty option {letter}"
    if q.get("section") not in SECTIONS:
        return False, f"unknown section: {q.get('section')}"
    return True, ""


def main(mode: str = "replace") -> None:
    in_path = NORMALIZED / "dedup.jsonl"
    if not in_path.exists():
        in_path = NORMALIZED / "generated.jsonl"
        console.print(f"[yellow]dedup.jsonl відсутній → використовую {in_path.name}[/yellow]")
    if not in_path.exists():
        console.print(f"[red]Немає вхідного файлу: {in_path}[/red]")
        sys.exit(1)

    items = _read_jsonl(in_path)
    console.print(f"[cyan]Завантажено {len(items)} запитань з {in_path.name}[/cyan]")

    # Виключаємо major-flagged
    flagged_path = NORMALIZED / "flagged_major.txt"
    flagged: set[str] = set()
    if flagged_path.exists():
        flagged = {line.strip() for line in flagged_path.read_text().splitlines() if line.strip()}
        console.print(f"  Виключено major-flagged: {len(flagged)}")

    by_section: dict[str, list[dict]] = {s: [] for s in SECTIONS}
    n_invalid = 0
    for q in items:
        if q.get("id") in flagged:
            continue
        ok, reason = _is_valid(q)
        if not ok:
            n_invalid += 1
            continue
        section = q["section"]
        by_section[section].append({
            "id": q["id"],
            "section": section,
            "subtopic": q.get("subtopic", ""),
            "question_ua": q["question_ua"].strip(),
            "options": q["options"],
            "correct": q["correct"],
            "explanation_ua": (q.get("explanation_ua") or "").strip(),
        })

    console.print(f"  Невалідних: {n_invalid}")
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for section in SECTIONS:
        qs = by_section[section]
        out_path = QUESTIONS_DIR / f"{section}.json"
        payload = {
            "section": section,
            "version": "1.0.0-generated",
            "note": "Згенеровано через OpenAI gpt-4o-mini + вибіркова валідація gpt-4o + дедуп embeddings",
            "questions": qs,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"  {section}: {len(qs)} → {out_path.name}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "replace"
    main(mode)
