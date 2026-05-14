"""
Генерація запитань КРОК-2 Стоматологія через OpenAI API.

Два режими:
  python -m scripts.generate smoke         — синхронний тест на 1 підтемі (5 запитань)
  python -m scripts.generate build-batch   — побудувати JSONL для Batch API
  python -m scripts.generate submit-batch  — завантажити JSONL і запустити батч
  python -m scripts.generate check         — перевірити статус останнього батчу
  python -m scripts.generate fetch         — завантажити результат батчу і парсити в JSONL

Цільова модель для масової генерації: gpt-4o-mini (Batch API, structured outputs).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

from scripts.curriculum import CURRICULUM, Subtopic, section_target_counts
from scripts.openai_schemas import QuestionBatch

# ---------------------------------------------------------------------------
# Налаштування
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "scripts" / "prompts"
DATA = ROOT / "data"
NORMALIZED = DATA / "normalized"
BATCH_DIR = DATA / "batch_results"

MODEL_BULK = "gpt-4o-mini"
MODEL_VALIDATE = "gpt-4o"

QUESTIONS_PER_REQUEST = 8

DEFAULT_TOTAL = 5000
DEFAULT_PROPORTIONS = {
    "general": 0.20,
    "therapeutic": 0.20,
    "surgical": 0.15,
    "orthopedic": 0.15,
    "pediatric_therapeutic": 0.10,
    "pediatric_surgical": 0.08,
    "orthodontic": 0.12,
}

console = Console()


def _load_env() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        console.print("[red]OPENAI_API_KEY не знайдено. Перевір .env у корені проекту.[/red]")
        sys.exit(1)
    return key


def _load_system_prompt() -> str:
    return (PROMPTS / "generate.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Планування генерації
# ---------------------------------------------------------------------------

@dataclass
class GenerationJob:
    section: str
    subtopic_key: str
    subtopic_label: str
    angles: tuple[str, ...]
    iteration: int          # 0..N-1, унікалізує запит, заохочує різноманіття
    count: int              # скільки запитань треба у цьому виклику

    @property
    def custom_id(self) -> str:
        return f"{self.section}__{self.subtopic_key}__iter{self.iteration:03d}"


def plan_jobs(total: int = DEFAULT_TOTAL,
              proportions: dict[str, float] = DEFAULT_PROPORTIONS,
              questions_per_request: int = QUESTIONS_PER_REQUEST) -> list[GenerationJob]:
    """
    Розподіляє total запитань між розділами за пропорціями, а далі рівномірно
    між підтемами розділу. Кожен запит до API виробляє questions_per_request запитань.
    """
    section_totals = section_target_counts(total, proportions)
    jobs: list[GenerationJob] = []

    for section, subtopics in CURRICULUM.items():
        section_total = section_totals.get(section, 0)
        if section_total <= 0 or not subtopics:
            continue
        # Розподіляємо запитання порівну між підтемами
        base = section_total // len(subtopics)
        remainder = section_total - base * len(subtopics)
        per_subtopic = [base + (1 if i < remainder else 0) for i in range(len(subtopics))]
        for st, q_count in zip(subtopics, per_subtopic):
            iterations = (q_count + questions_per_request - 1) // questions_per_request
            for i in range(iterations):
                this_batch = min(questions_per_request, q_count - i * questions_per_request)
                jobs.append(GenerationJob(
                    section=section,
                    subtopic_key=st.key,
                    subtopic_label=st.label_ua,
                    angles=st.angles,
                    iteration=i,
                    count=this_batch,
                ))
    return jobs


def _user_prompt_for(job: GenerationJob) -> str:
    angle = job.angles[job.iteration % len(job.angles)] if job.angles else ""
    return (
        f"Згенеруй {job.count} запитань КРОК-2 Стоматологія (5 курс) "
        f"з розділу «{_section_label_ua(job.section)}», підтема «{job.subtopic_label}»"
        + (f", акцент: «{angle}»" if angle else "")
        + ". Усі запитання різні (різні пацієнти, різні аспекти підтеми). "
        + f"У JSON-відповіді поле subtopic = «{job.subtopic_key}» для всіх запитань."
    )


def _section_label_ua(section: str) -> str:
    labels = {
        "general": "Загальний медичний профіль",
        "therapeutic": "Терапевтична стоматологія",
        "surgical": "Хірургічна стоматологія",
        "orthopedic": "Ортопедична стоматологія",
        "pediatric_therapeutic": "Дитяча терапевтична стоматологія",
        "pediatric_surgical": "Дитяча хірургічна стоматологія",
        "orthodontic": "Ортодонтія",
    }
    return labels.get(section, section)


# ---------------------------------------------------------------------------
# Smoke-тест (синхронний)
# ---------------------------------------------------------------------------

def cmd_smoke() -> None:
    """Один синхронний виклик: одна підтема, 5 запитань. Сейв у data/normalized/smoke.jsonl."""
    _load_env()
    client = OpenAI()
    system_prompt = _load_system_prompt()

    section = "therapeutic"
    subtopic = CURRICULUM[section][3]  # pulpitis
    job = GenerationJob(
        section=section,
        subtopic_key=subtopic.key,
        subtopic_label=subtopic.label_ua,
        angles=subtopic.angles,
        iteration=0,
        count=QUESTIONS_PER_REQUEST,
    )
    console.print(f"[cyan]Smoke test:[/cyan] section={section}, subtopic={subtopic.key}, count={job.count}")

    resp = client.beta.chat.completions.parse(
        model=MODEL_BULK,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _user_prompt_for(job)},
        ],
        response_format=QuestionBatch,
        temperature=0.8,
    )
    batch = resp.choices[0].message.parsed
    if batch is None:
        console.print("[red]Не вдалося розпарсити відповідь.[/red]")
        sys.exit(1)

    console.print(f"[green]Отримано {len(batch.questions)} запитань[/green]")
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    out_path = NORMALIZED / "smoke.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(batch.questions, 1):
            record = {
                "id": f"smoke-{section}-{subtopic.key}-{i:03d}",
                "section": section,
                "subtopic": q.subtopic or subtopic.key,
                "question_ua": q.question_ua,
                "options": q.options.model_dump(),
                "correct": q.correct,
                "explanation_ua": q.explanation_ua,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(f"[green]Збережено в {out_path}[/green]")
    # Показуємо перший приклад
    first = batch.questions[0]
    console.print("\n[bold]Приклад #1:[/bold]")
    console.print(first.question_ua)
    for letter, text in first.options.model_dump().items():
        marker = "✓" if letter == first.correct else " "
        console.print(f"  {marker} {letter}: {text}")
    console.print(f"[dim]Пояснення:[/dim] {first.explanation_ua}")


# ---------------------------------------------------------------------------
# Batch API — побудова JSONL
# ---------------------------------------------------------------------------

def cmd_build_batch(total: int = DEFAULT_TOTAL) -> None:
    """Будуємо JSONL з усіма запитами для Batch API і записуємо в data/batch_results/batch_input.jsonl."""
    _load_env()
    system_prompt = _load_system_prompt()
    jobs = plan_jobs(total=total)
    console.print(f"[cyan]Заплановано {len(jobs)} запитів, очікувано ~{sum(j.count for j in jobs)} запитань[/cyan]")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    input_path = BATCH_DIR / "batch_input.jsonl"
    schema = QuestionBatch.model_json_schema()
    # OpenAI structured outputs вимагає additionalProperties: false
    schema = _strict_json_schema(schema)

    with input_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            request = {
                "custom_id": job.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_BULK,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": _user_prompt_for(job)},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "QuestionBatch",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                    "temperature": 0.85,
                    "max_tokens": 8192,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    console.print(f"[green]Готово: {input_path}[/green]")
    console.print(f"  Запитів: {len(jobs)}")


def _strict_json_schema(schema: dict) -> dict:
    """OpenAI structured outputs strict mode:
    - additionalProperties: false + required для всіх властивостей у object
    - $ref не може мати сусідніх ключових слів (description, title, тощо)
    """
    def clean_ref_siblings(node: dict) -> None:
        """Якщо node має $ref, видалити description/title/default та інші ключі."""
        if "$ref" in node:
            keep = {"$ref"}
            for k in list(node.keys()):
                if k not in keep:
                    del node[k]

    def visit(node):
        if not isinstance(node, dict):
            return
        # У об'єктів — required + additionalProperties: false
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
            props = node.get("properties", {}) or {}
            if props:
                node["required"] = list(props.keys())
            for v in props.values():
                if isinstance(v, dict):
                    clean_ref_siblings(v)
                    visit(v)
        if "items" in node and isinstance(node["items"], dict):
            clean_ref_siblings(node["items"])
            visit(node["items"])
        for key in ("anyOf", "oneOf", "allOf"):
            if key in node:
                for v in node[key]:
                    if isinstance(v, dict):
                        clean_ref_siblings(v)
                        visit(v)
        if "$defs" in node:
            for v in node["$defs"].values():
                visit(v)
        if "definitions" in node:
            for v in node["definitions"].values():
                visit(v)
    visit(schema)
    return schema


# ---------------------------------------------------------------------------
# Batch API — запуск
# ---------------------------------------------------------------------------

def cmd_submit_batch() -> None:
    """Завантажує JSONL і створює батч."""
    _load_env()
    client = OpenAI()
    input_path = BATCH_DIR / "batch_input.jsonl"
    if not input_path.exists():
        console.print(f"[red]Файл {input_path} не існує. Спочатку запусти build-batch.[/red]")
        sys.exit(1)
    console.print(f"[cyan]Завантажую {input_path} ({input_path.stat().st_size // 1024} KB)...[/cyan]")
    file = client.files.create(file=input_path.open("rb"), purpose="batch")
    console.print(f"  File ID: {file.id}")
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": "krok2-dent", "kind": "generate"},
    )
    console.print(f"[green]Створено батч {batch.id}, статус: {batch.status}[/green]")
    (BATCH_DIR / "current_batch.txt").write_text(batch.id, encoding="utf-8")
    (BATCH_DIR / "current_batch.json").write_text(batch.model_dump_json(indent=2), encoding="utf-8")


def _current_batch_id() -> str:
    p = BATCH_DIR / "current_batch.txt"
    if not p.exists():
        console.print("[red]Немає current_batch.txt. Спочатку submit-batch.[/red]")
        sys.exit(1)
    return p.read_text().strip()


def cmd_check() -> None:
    _load_env()
    client = OpenAI()
    batch_id = _current_batch_id()
    b = client.batches.retrieve(batch_id)
    console.print(f"[cyan]Batch {batch_id}[/cyan]")
    console.print(f"  Status: [bold]{b.status}[/bold]")
    rc = b.request_counts
    console.print(f"  Requests: total={rc.total}, completed={rc.completed}, failed={rc.failed}")
    if b.errors:
        console.print(f"  Errors: {b.errors}")
    if b.status == "completed":
        console.print(f"  Output file: {b.output_file_id}")
        if b.error_file_id:
            console.print(f"  Error file: {b.error_file_id}")


def cmd_fetch() -> None:
    """Завантажує результат батчу, парсить структуровані відповіді в data/normalized/generated.jsonl."""
    _load_env()
    client = OpenAI()
    batch_id = _current_batch_id()
    b = client.batches.retrieve(batch_id)
    if b.status != "completed":
        console.print(f"[red]Батч ще не завершено: {b.status}[/red]")
        sys.exit(1)
    if not b.output_file_id:
        console.print("[red]Output file ID відсутній.[/red]")
        sys.exit(1)

    raw_path = BATCH_DIR / "batch_output.jsonl"
    content = client.files.content(b.output_file_id).read()
    raw_path.write_bytes(content)
    console.print(f"[green]Збережено raw output: {raw_path} ({len(content)//1024} KB)[/green]")

    out_path = NORMALIZED / "generated.jsonl"
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    n_questions, n_errors = 0, 0
    with raw_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            custom_id = entry.get("custom_id", "")
            response = entry.get("response", {})
            if response.get("status_code") != 200:
                n_errors += 1
                continue
            body = response.get("body", {})
            choices = body.get("choices", [])
            if not choices:
                n_errors += 1
                continue
            msg = choices[0].get("message", {})
            content_str = msg.get("content")
            if not content_str:
                n_errors += 1
                continue
            try:
                parsed = json.loads(content_str)
                for i, q in enumerate(parsed.get("questions", []), 1):
                    # Унормуємо
                    section = custom_id.split("__")[0] if custom_id else "unknown"
                    record = {
                        "id": f"gen-{custom_id}-q{i:02d}",
                        "section": section,
                        "subtopic": q.get("subtopic", ""),
                        "question_ua": q.get("question_ua", ""),
                        "options": q.get("options", {}),
                        "correct": q.get("correct", ""),
                        "explanation_ua": q.get("explanation_ua", ""),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_questions += 1
            except Exception as e:
                console.print(f"[yellow]Помилка парсингу для {custom_id}: {e}[/yellow]")
                n_errors += 1
    console.print(f"[green]Витягнуто {n_questions} запитань, помилок {n_errors}, файл: {out_path}[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        console.print("Usage: python -m scripts.generate [smoke|build-batch|submit-batch|check|fetch]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "smoke":
        cmd_smoke()
    elif cmd == "build-batch":
        total = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOTAL
        cmd_build_batch(total)
    elif cmd == "submit-batch":
        cmd_submit_batch()
    elif cmd == "check":
        cmd_check()
    elif cmd == "fetch":
        cmd_fetch()
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
