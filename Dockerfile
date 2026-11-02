FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY data data
COPY prompts prompts
COPY rules.yaml .
ENV RELEASELENS_DB=/app/data_runtime/releaselens.db
ENV RELEASELENS_CHROMA=/app/data_runtime/chroma_db
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
