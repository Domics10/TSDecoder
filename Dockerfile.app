# Immagine del servizio di previsione (solo inferenza). Per l'addestramento c'è Dockerfile.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=5002
WORKDIR /home/app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

EXPOSE 5002
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT} -w 1 --timeout 120 app:app"]
