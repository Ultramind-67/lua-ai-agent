import subprocess
import tempfile
import os
import re
from typing import TypedDict, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import json

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Инициализация LLM
llm = ChatOllama(
    model="qwen2.5-coder:7b",
    base_url=OLLAMA_HOST,
    temperature=0.1,
    num_ctx=4096,
    num_predict=256, # Жесткий лимит хакатона
)

class AgentState(TypedDict):
    prompt: str
    context: str
    template_used: str
    code: str
    errors: str
    iterations: int


def retrieve_template(state: AgentState) -> Dict:
    """Умный поиск шаблонов через Keyword Scoring (вместо жестких if-elif)."""
    user_prompt = state["prompt"].lower()
    best_template = ""

    base_dir = os.path.dirname(__file__)
    templates_path = os.path.join(base_dir, "templates", "templates.json")

    try:
        with open(templates_path, "r", encoding="utf-8") as f:
            templates = json.load(f)

        max_score = 0
        best_match = None

        # Проходим по всем шаблонам и считаем "очки"
        for tpl in templates:
            score = 0
            for keyword in tpl["keywords"]:
                # Проверяем, есть ли ключевое слово в запросе пользователя
                if keyword in user_prompt:
                    score += 1

            # Находим шаблон с максимальным совпадением
            if score > max_score:
                max_score = score
                best_match = tpl

        # Если нашли совпадения, отдаем шаблон как БАЗОВУЮ структуру, но разрешаем расширять
        if max_score >= 2 and best_match:
            print(f"DEBUG: RAG сработал! Найден шаблон: {best_match['id']}")
            best_template = (
                f"<instruction>\n"
                f"MANDATORY BASELINE TEMPLATE. You MUST use the core algorithm and platform functions (e.g. string.sub) shown in the template below.\n"
                f"HOWEVER, you are allowed to EXTEND and MODIFY this logic (add conditions, loops, or change variable names) IF the user's specific task requires it.\n"
                f"</instruction>\n"
                f"<template>\n{best_match['code']}\n</template>"
            )

    except Exception as e:
        print(f"Error loading templates: {e}")

    return {"template_used": best_template}


def generate_code(state: AgentState) -> Dict:
    """Генерация кода с учетом LowCode специфики и лимита в 256 токенов."""
    user_prompt = state["prompt"]
    context_data = state.get("context", "")
    errors = state.get("errors", "")
    template = state.get("template_used", "")

    # Отключаем RAG-шаблон при доработке кода, чтобы он не перебивал новые команды
    if context_data:
        template = ""

    # GUARDRAIL: защита от доработки "пустоты"
    if not context_data:
        prompt_lower = user_prompt.lower().strip()
        # Глаголы, которые обычно означают доработку существующего кода
        action_verbs = ["добавь", "измени", "удали", "исправь", "замени", "перепиши", "допиши", "сделай проверку", "убери", "не очищай"]
        # Указатели на существующий код
        pointers = ["в этот", "в данный", "текущий", "к этому", "сюда", "в код", "в функцию", "в массив"]

        # Если запрос начинается с глагола доработки ИЛИ содержит указатель
        is_modification = any(prompt_lower.startswith(v) for v in action_verbs) or \
                          any(p in prompt_lower for p in pointers)

        if is_modification:
            return {
                "code": "-- Пожалуйста, предоставьте исходный код для доработки.",
                "iterations": state.get("iterations", 0) + 1
            }

    # Если запрос слишком короткий, возвращаем вопрос-комментарий, чтобы не ломать API
    if len(user_prompt.split()) < 3:
        return {
            "code": "-- Пожалуйста, уточните задачу более подробно.",
            "iterations": state.get("iterations", 0) + 1
        }

    # Системный промпт (УЛУЧШЕНО ПРАВИЛО 7)
    system_msg = (
        "You are an expert Lua 5.5 developer for a LowCode platform.\n"
        "STRICT RULES:\n"
        "1. ALL runtime variables are in 'wf.vars' (e.g., wf.vars.my_array).\n"
        "2. To create a new array, strictly use '_utils.array.new()'.\n"
        "3. CONCISENESS: You have a 256 token limit. Use short variable names. EXCEPTION: If a <template> is provided, IGNORE conciseness and write the full logic from the template.\n"
        "4. Output ONLY VALID LUA CODE. No markdown tags like ```lua.\n"
        "5. ALWAYS return the final result using the 'return' statement.\n"
        "6. SCOPE LIMIT: You write SMALL LowCode snippets. If the prompt asks for a massive system, ONLY output: '-- Задача слишком объемна для одного скрипта.'\n"
        "7. MODIFYING CODE: If <context> is provided, it contains the OLD code. You MUST apply the user's 'Task' to this OLD code and output the NEW modified code. DO NOT just repeat the old code.\n"
        "8. NEVER create global functions. Always use 'local function name()' to pass static analysis.\n"
        "9. STATIC ANALYSIS: Name unused variables as '_' (e.g., 'for _, v in ipairs'). NEVER write '_' as a standalone statement on an empty line.\n"
        "10. Failure to use '_' for unused variables will break static analysis."
    )

    # Собираем финальный промпт
    final_user_content = ""

    if template:
        final_user_content += f"{template}\n\n"

    final_user_content += f"Task: {user_prompt}\n"

    if context_data:
        final_user_content += f"\n<context>\n{context_data}\n</context>\n"

    if errors:
        final_user_content += f"\nFIX THIS SYNTAX ERROR in previous attempt:\n{errors}\n"

    # Отправляем в LLM
    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=final_user_content)
    ])

    # Очистка вывода от возможных маркдаун-тегов
    code = response.content.replace("```lua", "").replace("```", "").strip()

    # Фикс для частой проблемы обрыва генерации из-за 256 токенов
    if "end" not in code and ("for " in code or "if " in code or "function " in code):
        code += "\nend -- auto-fixed missing end due to token limits"

    return {"code": code, "iterations": state.get("iterations", 0) + 1}


def validate_code(state: AgentState) -> Dict:
    """Многоуровневая валидация: Синтаксис (luac) + Статический анализ (luacheck)"""
    code = state["code"]

    if code.startswith("-- Пожалуйста") or code.startswith("-- Внимание") or code.startswith("-- Уточните"):
        return {"errors": ""}

    clean_code = re.sub(r'^lua\{', '', code)
    clean_code = re.sub(r'\}lua$', '', clean_code).strip()

    with tempfile.NamedTemporaryFile(suffix=".lua", delete=False) as tmp:
        tmp.write(clean_code.encode('utf-8'))
        tmp_path = tmp.name

    try:
        # Уровень 1: Строгая синтаксическая проверка
        syntax_check = subprocess.run(['luac', '-p', tmp_path], capture_output=True, text=True, timeout=5)
        if syntax_check.returncode != 0:
            err_msg = syntax_check.stderr.replace(tmp_path, 'script.lua')

            # GUARDRAIL: Если модель написала одиночный '_', подсказываем ей, как это исправить
            if "'=' expected near 'end'" in err_msg and "_" in code:
                err_msg += "\nHINT FROM LINTER: You wrote a standalone '_'. In Lua, you must assign it, e.g., 'local _ = something' or just leave the loop empty."

            return {"errors": f"Syntax Error: {err_msg}"}

        # Уровень 2: Статический анализ (luacheck)
        # Разрешаем глобальные переменные платформы: wf, _utils, os, string, math, table
        lint_check = subprocess.run([
            'luacheck', tmp_path,
            '--globals', 'wf', '_utils', 'os', 'string', 'math', 'table',
            '--no-max-line-length',
            '--formatter', 'plain'
        ], capture_output=True, text=True, timeout=5)

        # luacheck возвращает 0 если всё идеально, или варнинги/ошибки
        if lint_check.returncode > 0:
            # Берем только суть ошибки, без путей к временному файлу
            lint_errors = lint_check.stdout.replace(tmp_path, 'script.lua')
            return {
                "errors": f"Static Analysis Warning/Error: {lint_errors}\nПожалуйста, исправь эти логические ошибки."}

        return {"errors": ""}
    except Exception as e:
        return {"errors": str(e)}
    finally:
        os.remove(tmp_path)

def router(state: AgentState) -> str:
    if state.get("errors") and state["iterations"] < 3:
        return "generate_code"
    return "end_success"

# Сборка графа
workflow = StateGraph(AgentState)

workflow.add_node("retrieve_template", retrieve_template)
workflow.add_node("generate_code", generate_code)
workflow.add_node("validate_code", validate_code)

workflow.set_entry_point("retrieve_template")
workflow.add_edge("retrieve_template", "generate_code")
workflow.add_edge("generate_code", "validate_code")

workflow.add_conditional_edges(
    "validate_code",
    router,
    {"generate_code": "generate_code", "end_success": END}
)

app_agent = workflow.compile()