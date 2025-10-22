@echo off
echo ====================================
echo Starting ViMedNER API Server
echo ====================================
echo.

cd /d D:\Project\NLP

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting FastAPI server on port 8001...
uvicorn src.web.backend.api_server:app --host 127.0.0.1 --port 8001 --reload

pause