FROM python:3.10-slim

# Установка компилятора Lua для валидации кода агентом
RUN apt-get update && apt-get install -y lua5.4 luarocks && luarocks install luacheck && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY ./app ./app

# Запуск FastAPI сервера
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]