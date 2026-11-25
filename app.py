"""Servizio HTTP di previsione: solo calcolo, nessuna validazione complessa.

POST /predict
  ingresso: {"time_series": [{"timestamp", "value"}], "model", "request_id", "auth", "horizon"?}
  uscita:   {"predictions": [{"timestamp", "value"}], "request_id", "auth"}
GET /status
GET /models  -> [{"name", "min_points"}]
"""
import os
import sys

from flask import Flask, jsonify, request

from forecasters import build_registry, next_timestamps

DEFAULT_HORIZON = int(os.environ.get("DEFAULT_HORIZON", "10"))

app = Flask(__name__)
FORECASTERS = build_registry()


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
    payload = request.get_json(silent=True) or {}
    series = payload.get("time_series") or []
    forecaster = FORECASTERS.get(payload.get("model"))
    horizon = payload.get("horizon", DEFAULT_HORIZON)

    problem = first_problem([
        (lambda: forecaster is None,
         lambda: f"Modello sconosciuto: {payload.get('model')!r}. Disponibili: {', '.join(sorted(FORECASTERS))}."),
        (lambda: len(series) < forecaster.min_points,
         lambda: f"Il modello {payload.get('model')} richiede almeno {forecaster.min_points} punti, ricevuti {len(series)}."),
        (lambda: not isinstance(horizon, int) or horizon < 1,
         lambda: "L'orizzonte deve essere un intero positivo."),
    ])
    if problem:
        return jsonify({"errore": problem}), 400

    try:
        values = forecaster([point["value"] for point in series], horizon)
        timestamps = next_timestamps([point["timestamp"] for point in series], horizon)
    except Exception as exc:  # guasto di calcolo: si logga e si risponde 500
        print(f"Errore di previsione ({payload.get('request_id')}): {exc!r}", file=sys.stderr)
        return jsonify({"errore": "Errore nel calcolo della previsione."}), 500

    return jsonify({
        "predictions": [{"timestamp": ts, "value": v} for ts, v in zip(timestamps, values)],
        "request_id": payload.get("request_id"),
        "auth": payload.get("auth"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5002")))
