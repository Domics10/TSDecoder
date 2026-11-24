# TSDecoder

Transformer decoder per previsioni su serie temporali (self-attention multi-head, skip connection,
layer norm, dropout), addestrato sulle temperature medie mensili in `data/`.

- `main.py`: addestramento e validazione (split 80/20, finestra di 90 passi; iperparametri in `model/model.json`).
- `saved_models/`: i pesi addestrati.
- `app.py` + `forecasters.py`: servizio HTTP di sola inferenza.

## Servizio di previsione

```bash
docker build -f Dockerfile.app -t tsdecoder-app .
docker run -p 5002:5002 tsdecoder-app
```

Variabili d'ambiente: `PORT` (default 5002), `DEFAULT_HORIZON` (punti da prevedere, default 10).

Modelli disponibili:
- `linear`: regressione lineare, da 2 punti in su;
- `TSDecoder`: il decoder addestrato, servono almeno 90 punti (la finestra si riporta nella scala dei
  dati di addestramento e le previsioni tornano nella scala dell'ingresso).

`POST /predict`

```bash
curl -s localhost:5002/predict -H 'Content-Type: application/json' -d '{
  "model": "linear",
  "request_id": "esempio-1",
  "auth": {"username": "utente"},
  "time_series": [
    {"timestamp": "2026-01-01", "value": 10},
    {"timestamp": "2026-01-02", "value": 12},
    {"timestamp": "2026-01-03", "value": 14}
  ],
  "horizon": 3
}'
```

Risposta: `{"predictions": [{"timestamp", "value"}], "request_id", "auth"}`, con i timestamp che
proseguono il passo della serie. Un modello sconosciuto o troppi pochi punti danno 400 con il motivo.

`GET /status`: `{"status": "ok", "models": [...]}`.
