# Immagine di addestramento e validazione (main.py). Per il servizio di previsione c'è Dockerfile.app.
#   docker build -t tsdecoder .
#   docker run --rm tsdecoder                          # validazione sui pesi salvati
#   docker run --rm tsdecoder python main.py --addestra
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /home/app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
