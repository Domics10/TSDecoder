"""Servizio HTTP di previsione: solo calcolo, nessuna validazione complessa.

POST /predict
  ingresso: {"time_series": [{"timestamp", "value"}], "model", "request_id", "auth", "horizon"?, "env"?}
  "env": variabili del container impostate dal client (DEFAULT_HORIZON, SEASON_PERIOD); valgono per questa richiesta.
  orizzonte: "horizon", altrimenti DEFAULT_HORIZON di "env", altrimenti quella del processo; fra 1 e 1000.
  uscita:   {"predictions": [{"timestamp", "value"}], "request_id", "auth"}
GET /status
GET /models  -> [{"name", "min_points"}]
POST /federated
  ingresso: {"target": [{"timestamp", "value"}], "clients": [[{"timestamp", "value"}], …], "order"?: 5, "horizon"?, "env"?}
  uscita:   {"predictions": [...], "coefficients": [...], "participants": [{"samples", "coefficients"}]}
POST /metrics
  ingresso: {"forecast": [numeri], "actual": [numeri], "history"?: [numeri], "season"?: intero}
  uscita:   errori della previsione (me, mae, rmse, mape, mpe, smape, mase, corr, minmax, acf1, points)
"""
import os
import sys

from flask import Flask, jsonify, request

from forecasters import build_registry, federated_ar, next_timestamps
from model.metrics import ForecastErrors

DEFAULT_HORIZON = os.environ.get("DEFAULT_HORIZON", "10")
MAX_HORIZON = 1000

app = Flask(__name__)
FORECASTERS = build_registry()


def read_payload():
    """Corpo JSON della richiesta; un corpo che non è un oggetto vale come vuoto."""
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def request_env(payload):
    """Variabili mandate con la richiesta ("env"); se non sono un oggetto, nessuna."""
    return payload.get("env") if isinstance(payload.get("env"), dict) else {}


def setting(payload, name, default):
    """Valore di una variabile mandata con la richiesta ("env"), o `default` se manca
    (per DEFAULT_HORIZON il default viene dall'ambiente del processo)."""
    return request_env(payload).get(name, default)


def as_horizon(value):
    """Intero fra 1 e MAX_HORIZON, o None se il valore non lo è."""
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= MAX_HORIZON and not isinstance(value, bool) else None


def first_problem(checks):
    """Primo motivo di rifiuto fra le coppie (condizione, messaggio), valutate in ordine."""
    return next((message() for condition, message in checks if condition()), None)


@app.get("/status")
def status():
    return jsonify({"status": "ok", "models": sorted(FORECASTERS)})


@app.get("/models")
def models():
    return jsonify([{"name": name, "min_points": f.min_points} for name, f in sorted(FORECASTERS.items())])


@app.post("/predict")
def predict():
    payload = read_payload()
    series = payload.get("time_series") or []
    forecaster = FORECASTERS.get(payload.get("model"))
    horizon = as_horizon(payload.get("horizon", setting(payload, "DEFAULT_HORIZON", DEFAULT_HORIZON)))

    problem = first_problem([
        (lambda: forecaster is None,
         lambda: f"Modello sconosciuto: {payload.get('model')!r}. Disponibili: {', '.join(sorted(FORECASTERS))}."),
        (lambda: len(series) < forecaster.min_points,
         lambda: f"Il modello {payload.get('model')} richiede almeno {forecaster.min_points} punti, ricevuti {len(series)}."),
        (lambda: horizon is None,
         lambda: f"L'orizzonte (horizon o DEFAULT_HORIZON) deve essere un intero fra 1 e {MAX_HORIZON}."),
    ])
    if problem:
        return jsonify({"errore": problem}), 400

    try:  # prima le date: una richiesta impossibile fallisce prima del calcolo
        timestamps = next_timestamps([point["timestamp"] for point in series], horizon)
        values = forecaster([point["value"] for point in series], horizon, request_env(payload))
    except ValueError as exc:  # impostazione non valida per quel modello: si corregge
        return jsonify({"errore": f"{payload.get('model')}: {exc}."}), 400
    except Exception as exc:  # guasto di calcolo: si logga e si risponde 500
        print(f"Errore di previsione ({payload.get('request_id')}): {exc!r}", file=sys.stderr)
        return jsonify({"errore": "Errore nel calcolo della previsione."}), 500

    return jsonify({
        "predictions": [{"timestamp": ts, "value": v} for ts, v in zip(timestamps, values)],
        "request_id": payload.get("request_id"),
        "auth": payload.get("auth"),
    })


@app.post("/federated")
def federated():
    payload = read_payload()
    target = payload.get("target") or []
    clients = payload.get("clients") or []
    horizon = as_horizon(payload.get("horizon", setting(payload, "DEFAULT_HORIZON", DEFAULT_HORIZON)))
    order = as_horizon(payload.get("order", 5))
    problem = first_problem([
        (lambda: not target, lambda: "Serve la serie bersaglio (target)."),
        (lambda: horizon is None or order is None,
         lambda: f"Orizzonte e ordine devono essere interi fra 1 e {MAX_HORIZON}."),
    ])
    if problem:
        return jsonify({"errore": problem}), 400
    try:
        timestamps = next_timestamps([p["timestamp"] for p in target], horizon)
        result = federated_ar([p["value"] for p in target], [[p["value"] for p in c] for c in clients], order, horizon)
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"errore": f"Modello federato: {exc}."}), 400
    return jsonify({
        "predictions": [{"timestamp": ts, "value": v} for ts, v in zip(timestamps, result["values"])],
        "coefficients": result["coefficients"],
        "participants": result["participants"],
    })


@app.post("/metrics")
def metrics():
    payload = read_payload()
    try:
        errors = ForecastErrors(payload.get("forecast"), payload.get("actual"),
                                payload.get("history"), payload.get("season", 1))
    except (TypeError, ValueError) as exc:
        return jsonify({"errore": f"Dati non validi per il calcolo degli errori: {exc}"}), 400
    return jsonify(errors.summary())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5002")))
