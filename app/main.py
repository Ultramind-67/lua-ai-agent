from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import app_agent
from fastapi.responses import HTMLResponse
from typing import Optional

app = FastAPI(title="LocalScript API", version="1.0.0")


# Оставляем UI для демо и жюри
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("app/index.html", "r", encoding="utf-8") as f:
        return f.read()


# Строго по OpenAPI YAML
class GenerateRequest(BaseModel):
    prompt: str
    context: Optional[str] = ""


class GenerateResponse(BaseModel):
    code: str


@app.post("/generate", response_model=GenerateResponse)
def generate_script(request: GenerateRequest):
    # Инициализируем стейт
    initial_state = {
        "prompt": request.prompt,
        "context": request.context,  # Прокидываем контекст
        "template_used": "",
        "code": "",
        "errors": "",
        "iterations": 0
    }

    # Запускаем графового агента
    final_state = app_agent.invoke(initial_state)

    code_result = final_state.get("code", "")
    errors = final_state.get("errors", "")

    # Если после 3 попыток код все еще с ошибками, оборачиваем ошибку в Lua-комментарий,
    if errors and final_state["iterations"] >= 3:
        safe_error = errors.replace('\n', ' ')
        code_result = f"-- Внимание: Не удалось сгенерировать валидный код за 3 итерации.\n-- Последняя ошибка: {safe_error}\n\n{code_result}"

    return GenerateResponse(code=code_result)