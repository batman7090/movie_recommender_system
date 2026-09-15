FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home appuser
COPY --chown=appuser:appuser recommender ./recommender
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser app.py .
RUN mkdir /app/models /app/data && chown appuser:appuser /app/models /app/data
USER appuser
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
