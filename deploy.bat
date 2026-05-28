@echo off
echo ===================================================
echo Розгортання Agentic RAG системи за допомогою Docker
echo ===================================================

echo Перевірка наявності файлу .env...
if not exist ".env" (
    echo Файл .env не знайдено! Копіюємо з .env.example...
    copy .env.example .env
    echo Будь ласка, переконайтесь, що ви налаштували GROQ_API_KEY у файлі .env!
)

echo.
echo Збирання та запуск Docker-контейнерів...
docker-compose up --build -d

echo.
echo Система розгорнута!
echo API доступне за адресою: http://localhost:8000
echo Документація Swagger: http://localhost:8000/docs
echo ===================================================
pause
