"""
Валідація і дедуплікація згенерованих запитань.

  python -m scripts.validate dedup           — дедуп через OpenAI embeddings (text-embedding-3-small)
  python -m scripts.validate sample [N]      — вибірково валідувати N запитань через gpt-4o (default 10% від загальної кількості)

Вхід: data/normalized/generated.jsonl
Вихід: data/normalized/dedup.jsonl, data/normalized/validation.jsonl
"""
from __future__ import annotations

import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from tqdm import tqdm

from scripts.openai_schemas import ValidationVerdict

VALIDATE_CONCURRENCY = 8

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NORMALIZED = DATA / "normalized"
PROMPTS = ROOT / "scripts" / "prompts"

MODEL_VALIDATE = "gpt-5"
EMBED_MODEL = "text-embedding-3-small"
DEDUP_THRESHOLD = 0.92  # косинусна подібність вище → дублікат

console = Console()


def _load_env() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY не знайдено.[/red]")
        sys.exit(1)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        console.print(f"[red]Файл не існує: {path}[/red]")
        sys.exit(1)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Дедуплікація через OpenAI embeddings
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    return s  # embeddings вже нормалізовані OpenAI'єм


def _embed_batch(client: OpenAI, texts: list[str], batch_size: int = 256) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding", unit="batch"):
        chunk = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        embeddings.extend(d.embedding for d in resp.data)
    return embeddings


def cmd_dedup() -> None:
    _load_env()
    client = OpenAI()
    in_path = NORMALIZED / "generated.jsonl"
    out_path = NORMALIZED / "dedup.jsonl"

    items = _read_jsonl(in_path)
    console.print(f"[cyan]Завантажено {len(items)} запитань.[/cyan]")

    # Готуємо тексти: question + options (без correct, щоб не дублювали між собою через відповідь)
    def serialize(q: dict) -> str:
        opts = q.get("options", {})
        opts_str = " | ".join(f"{k}: {v}" for k, v in opts.items())
        return f"{q.get('question_ua', '')} | {opts_str}"

    # Усереди розділу шукаємо дублікати; між розділами не порівнюємо
    by_section: dict[str, list[int]] = {}
    for i, q in enumerate(items):
        by_section.setdefault(q.get("section", "unknown"), []).append(i)

    keep_mask = [True] * len(items)
    n_dups_total = 0

    for section, idxs in by_section.items():
        if len(idxs) < 2:
            continue
        console.print(f"[cyan]Розділ {section}: {len(idxs)} запитань[/cyan]")
        texts = [serialize(items[i]) for i in idxs]
        embeddings = _embed_batch(client, texts)
        # Жадібний дедуп: для кожного нового елемента порівнюємо з усіма попередніми, що залишилися
        kept_local: list[int] = []
        kept_embeds: list[list[float]] = []
        for local_pos, embed in enumerate(embeddings):
            is_dup = False
            for kept_embed in kept_embeds:
                if _cosine(embed, kept_embed) > DEDUP_THRESHOLD:
                    is_dup = True
                    break
            if is_dup:
                keep_mask[idxs[local_pos]] = False
                n_dups_total += 1
            else:
                kept_local.append(local_pos)
                kept_embeds.append(embed)

    kept = [q for q, m in zip(items, keep_mask) if m]
    _write_jsonl(out_path, kept)
    console.print(f"[green]Видалено дублікатів: {n_dups_total}, залишено: {len(kept)} → {out_path}[/green]")


# ---------------------------------------------------------------------------
# Валідація вибіркою gpt-4o
# ---------------------------------------------------------------------------

def _validate_prompt(q: dict) -> str:
    opts = q.get("options", {})
    opts_str = "\n".join(f"  {k}: {v}" for k, v in opts.items())
    return (
        f"Перевір це запитання КРОК-2 Стоматологія (5 курс).\n\n"
        f"Розділ: {q.get('section')}\n"
        f"Підтема: {q.get('subtopic')}\n\n"
        f"Запитання:\n{q.get('question_ua')}\n\n"
        f"Варіанти:\n{opts_str}\n\n"
        f"Маркована правильна відповідь: {q.get('correct')}\n"
        f"Пояснення: {q.get('explanation_ua', '')}"
    )


def cmd_sample(n_arg: str | None) -> None:
    _load_env()
    client = OpenAI()
    in_path = NORMALIZED / "dedup.jsonl"
    if not in_path.exists():
        in_path = NORMALIZED / "generated.jsonl"
        console.print(f"[yellow]dedup.jsonl відсутній, валідую generated.jsonl напряму[/yellow]")
    items = _read_jsonl(in_path)
    sample_size = int(n_arg) if n_arg else max(20, len(items) // 10)
    sample_size = min(sample_size, len(items))
    sample = random.sample(items, sample_size)
    console.print(f"[cyan]Валідую {sample_size} з {len(items)} запитань через {MODEL_VALIDATE}...[/cyan]")

    system_prompt = (PROMPTS / "validate.md").read_text(encoding="utf-8")

    def _validate_one(q: dict) -> dict | None:
        try:
            resp = client.beta.chat.completions.parse(
                model=MODEL_VALIDATE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _validate_prompt(q)},
                ],
                response_format=ValidationVerdict,
            )
            verdict = resp.choices[0].message.parsed
            if verdict is None:
                return None
            return {
                "id": q["id"],
                "section": q.get("section"),
                "subtopic": q.get("subtopic"),
                "is_correct": verdict.is_correct,
                "is_medically_sound": verdict.is_medically_sound,
                "severity": verdict.severity,
                "issues": verdict.issues,
            }
        except Exception as e:
            console.print(f"[yellow]Помилка для {q.get('id')}: {e}[/yellow]")
            return None

    results: list[dict] = []
    flagged_ids: list[str] = []
    severity_counts = {"none": 0, "minor": 0, "major": 0}

    with ThreadPoolExecutor(max_workers=VALIDATE_CONCURRENCY) as ex:
        futures = {ex.submit(_validate_one, q): q for q in sample}
        with tqdm(total=len(futures), desc="validate", unit="q") as pbar:
            for fut in as_completed(futures):
                record = fut.result()
                pbar.update(1)
                if record is None:
                    continue
                severity_counts[record["severity"]] += 1
                results.append(record)
                if record["severity"] == "major":
                    flagged_ids.append(record["id"])

    out_path = NORMALIZED / "validation.jsonl"
    _write_jsonl(out_path, results)
    console.print(f"\n[green]Звіт валідації → {out_path}[/green]")
    console.print(f"  none: {severity_counts['none']}, minor: {severity_counts['minor']}, major: {severity_counts['major']}")
    console.print(f"  Major rate: {severity_counts['major']/max(1, sample_size)*100:.1f}%")

    # Експортуємо id з major issues, щоб відкинути в export
    if flagged_ids:
        flagged_path = NORMALIZED / "flagged_major.txt"
        flagged_path.write_text("\n".join(flagged_ids), encoding="utf-8")
        console.print(f"  Major-flagged IDs → {flagged_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        console.print("Usage: python -m scripts.validate [dedup|sample [N]]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "dedup":
        cmd_dedup()
    elif cmd == "sample":
        cmd_sample(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
