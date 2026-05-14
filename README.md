# КРОК-2 Стоматологія — тренажер

Веб-додаток для підготовки до українського ліцензійного іспиту **КРОК-2 Стоматологія** (5-й курс). Імітує реальний іспит: 150 випадкових запитань за 200 хвилин з правильними пропорціями розділів, повний розбір з поясненнями в кінці.

🌐 **Live:** https://666alucard666.github.io/krok2-dent/

## Як запустити локально

```bash
cd docs
python3 -m http.server 8000
```

Відкрити: http://localhost:8000

## Структура

- `docs/` — статичний сайт (GitHub Pages)
  - `index.html`, `app.js`, `styles.css` — Alpine.js SPA
  - `config.json` — пропорції розділів за реальним КРОК-2
  - `questions/*.json` — база запитань по 7 розділах
- `scripts/` — Python-пайплайн генерації бази (OpenAI API)
- `data/` — сирі та проміжні дані (gitignored)

## Розділи

1. Загальний медичний профіль (~20%)
2. Терапевтична стоматологія (~20%)
3. Хірургічна стоматологія (~15%)
4. Ортопедична стоматологія (~15%)
5. Дитяча терапевтична стоматологія (~10%)
6. Дитяча хірургічна стоматологія (~8%)
7. Ортодонтія (~12%)

## Хостинг

Деплой на GitHub Pages автоматично при `git push` до `main` через `.github/workflows/pages.yml`.

## Пайплайн генерації

Стратегія: curriculum-driven генерація через OpenAI API. Замість парсингу публічних
джерел (які використовують JS-рендеринг — фрагільно) ми генеруємо нові клінічно
обґрунтовані запитання за детальним curriculum-древом КРОК-2 Стоматологія.

**Стек:**
- `gpt-4o-mini` — масова генерація через Batch API (1028 запитів → ~5000 запитань)
- `text-embedding-3-small` — семантична дедуплікація (cosine > 0.92)
- `gpt-4o` — вибіркова валідація 10% (medical соунд + правильність відповіді)
- Structured outputs через Pydantic (`scripts/openai_schemas.py`)

**Команди:**

```bash
cd /Users/dmitrovahnenko/krok2-dent
python3 -m venv .venv
source .venv/bin/activate
pip install openai pydantic python-dotenv rich tqdm

# Покладіть OPENAI_API_KEY у .env
echo "OPENAI_API_KEY=sk-..." > .env

# 1. Швидкий тест на 5 запитаннях
python -m scripts.generate smoke

# 2. Побудувати batch JSONL для повної бази
python -m scripts.generate build-batch 5000

# 3. Засабмітити batch до OpenAI
python -m scripts.generate submit-batch

# 4. Перевірити статус
python -m scripts.generate check

# 5. Коли completed → fetch
python -m scripts.generate fetch

# 6. Дедуп через embeddings
python -m scripts.validate dedup

# 7. Вибіркова валідація через gpt-4o (10%)
python -m scripts.validate sample

# 8. Експорт у docs/questions/
python -m scripts.export

# 9. Commit і push — GitHub Pages автоматично оновить сайт
git add docs/questions/ && git commit -m "Update questions" && git push
```

**Вартість:** ~$5-15 для повної бази 5000 запитань (Batch API 50% знижка + prompt caching).

**Curriculum:** 86 підтем у 7 розділах, кожна з 3-6 «кутами» (діагностика/лікування/ускладнення/тощо) для різноманіття. Див. `scripts/curriculum.py`.
