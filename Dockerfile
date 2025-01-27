# Используем базовый образ Python
FROM python:3.10.15-slim

# Устанавливаем PDM
RUN pip install pdm

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем pyproject.toml и pdm.lock для установки зависимостей
COPY pyproject.toml pdm.lock /app/

# Устанавливаем зависимости с помощью PDM
RUN pdm install --production

# Копируем исходный код
COPY src /app/src
COPY .env /app/.env

EXPOSE 14956

# Команда для запуска сервиса через Gunicorn
CMD ["pdm", "run", "gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.main:app", "--bind", "0.0.0.0:14956"]
