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
`DEFAULT_HORIZON` si può anche mandare con la richiesta, nel campo `env` (`{"env": {"DEFAULT_HORIZON": "5"}}`):
vince su quella del processo; `horizon` nella richiesta vince su tutte e due.

Modelli disponibili:
- `linear`: regressione lineare, da 2 punti in su;
- `seasonal`: ripete il valore di 12 passi prima (stagionale naive), da 12 punti in su;
- `TSDecoder`: il decoder addestrato, servono almeno 90 punti (la finestra si riporta nella scala dei
  dati di addestramento e le previsioni tornano nella scala dell'ingresso).

`GET /models`: `[{"name", "min_points"}]`, l'elenco qui sopra.

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

Test: `python -m unittest discover -s tests -t .` (nell'immagine di `Dockerfile.app`).

## Quanto è buono il modello (25 nov 2026)

Valutazione rolling-origin sul 20% finale della serie (84 origini, finestra 90, orizzonte 10 mesi),
errore assoluto medio in °C sui mesi 1..10:

| Metodo | MAE |
|---|---|
| stagionale (valore di 12 mesi prima) | 0,50 |
| regressione lineare sulla finestra | 0,63 |
| media della finestra | 0,65 |
| TSDecoder come lo usa il servizio (finestra riscalata) | 0,69 |
| TSDecoder sulla finestra grezza | 1,37 |

Il decoder funziona e segue la stagionalità, ma oggi non batte i metodi banali. `main.py` (92 passi
autoregressivi, dropout attivo) dà MAPE 40%: la previsione crolla dopo pochi mesi, come già nel 2023.
Va riaddestrato prima di preferirlo agli altri.
