"""Pydantic-моделі для structured outputs OpenAI."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class Options(BaseModel):
    A: str
    B: str
    C: str
    D: str
    E: str


class GeneratedQuestion(BaseModel):
    question_ua: str = Field(description="Текст клінічного запитання українською")
    options: Options = Field(description="П'ять варіантів відповіді")
    correct: Literal["A", "B", "C", "D", "E"] = Field(description="Літера правильного варіанту")
    explanation_ua: str = Field(description="Стисле пояснення 2-4 речення")
    subtopic: str = Field(description="Підтема, до якої належить запитання")


class QuestionBatch(BaseModel):
    """Кілька запитань на одну підтему — те, що OpenAI повертає за один виклик."""
    questions: list[GeneratedQuestion] = Field(description="Список згенерованих запитань")


class ValidationVerdict(BaseModel):
    is_correct: bool = Field(description="Чи правильне маркування правильної відповіді")
    is_medically_sound: bool = Field(description="Чи медично обґрунтоване питання")
    issues: list[str] = Field(description="Список проблем (порожній якщо все ок)")
    severity: Literal["none", "minor", "major"] = Field(description="Серйозність проблем")
