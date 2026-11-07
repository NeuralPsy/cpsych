FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[test]" --no-cache-dir
RUN python article/verify_numbers.py
CMD ["pytest", "-m", "not slow", "--tb=short", "-q"]
