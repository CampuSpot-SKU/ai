FROM python:3.11-slim
WORKDIR /app
COPY . .
# 의존성 목록은 pyproject.toml 하나로 관리 (requirements.txt 없음)
RUN pip install --no-cache-dir .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
