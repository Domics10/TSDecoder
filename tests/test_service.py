"""Test del servizio di previsione. Si lanciano dalla radice del repo:

    python -m unittest discover -s tests -t .
"""
import unittest

import numpy as np
import pandas as pd

from forecasters import (LinearForecaster, NaiveForecaster, SeasonalForecaster, TSDecoderForecaster,
                         federated_ar, fit_ar, next_timestamps, rescale)

DECODER = TSDecoderForecaster.load()


def wave(n, offset=0.0):
    """Serie di prova con stagionalità di 12 passi."""
    return (np.sin(np.arange(offset, offset + n) * 2 * np.pi / 12) + 27.5).tolist()


class RescaleTest(unittest.TestCase):
    def test_porta_gli_estremi_negli_estremi(self):
        np.testing.assert_allclose(rescale([0, 5, 10], (0, 10), (20, 30)), [20, 25, 30])

    def test_serie_costante_va_al_centro(self):
        np.testing.assert_allclose(rescale([3, 3], (3, 3), (0, 10)), [5, 5])


class LinearForecasterTest(unittest.TestCase):
    def test_prosegue_una_retta(self):
        np.testing.assert_allclose(LinearForecaster()([1, 3, 5], 3), [7, 9, 11])


class NaiveForecasterTest(unittest.TestCase):
    def test_ripete_l_ultimo_valore(self):
        self.assertEqual(NaiveForecaster()([1, 2, 7], 3), [7.0, 7.0, 7.0])


class FederatedTest(unittest.TestCase):
    def test_ar_ritrova_una_ricorrenza(self):
        # Serie che alterna: AR(1) con coefficiente negativo.
        coefficients, samples = fit_ar([1, -1, 1, -1, 1, -1, 1, -1], 1)
        self.assertLess(coefficients[0], -0.9)
        self.assertEqual(samples, 7)

    def test_fedavg_pesa_i_partecipanti_e_resta_nella_scala_del_bersaglio(self):
        wave = [np.sin(i / 2) for i in range(60)]
        target = [100 + 10 * v for v in wave]
        result = federated_ar(target, [wave, [3 * v for v in wave]], order=3, horizon=4)
        self.assertEqual(len(result["values"]), 4)
        self.assertEqual(len(result["participants"]), 3)
        self.assertTrue(all(85 < v < 115 for v in result["values"]))

    def test_serie_troppo_corte(self):
        with self.assertRaises(ValueError):
            federated_ar([1, 2, 3], [], order=5, horizon=2)


class SeasonalForecasterTest(unittest.TestCase):
    def test_ripete_l_ultimo_periodo(self):
        self.assertEqual(SeasonalForecaster(period=3)([9, 1, 2, 3], 5), [1, 2, 3, 1, 2])

    def test_periodo_dalle_variabili(self):
        self.assertEqual(SeasonalForecaster()([1, 2, 3, 4, 5], 4, {"SEASON_PERIOD": "2"}), [4, 5, 4, 5])
        with self.assertRaises(ValueError):
            SeasonalForecaster()([1, 2], 1, {"SEASON_PERIOD": "9"})

    def test_serve_un_periodo_intero(self):
        self.assertEqual(SeasonalForecaster().min_points, 12)


class NextTimestampsTest(unittest.TestCase):
    def test_passo_giornaliero(self):
        self.assertEqual(
            next_timestamps(["2026-01-01", "2026-01-02", "2026-01-03"], 2),
            ["2026-01-04T00:00:00Z", "2026-01-05T00:00:00Z"],
        )

    def test_passo_mensile_rispetta_i_mesi(self):
        self.assertEqual(
            next_timestamps(["2026-01-01", "2026-02-01", "2026-03-01"], 2),
            ["2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z"],
        )

    def test_due_punti_usano_la_loro_distanza(self):
        self.assertEqual(
            next_timestamps(["2026-01-01T00:00:00Z", "2026-01-01T06:00:00Z"], 1),
            ["2026-01-01T12:00:00Z"],
        )

    def test_giorni_di_borsa_saltano_il_fine_settimana(self):
        days = pd.bdate_range("2026-01-05", periods=15).strftime("%Y-%m-%d").tolist()
        days.remove("2026-01-19")  # una festività: la frequenza non si deduce più
        self.assertEqual(next_timestamps(days, 2), ["2026-01-26T00:00:00Z", "2026-01-27T00:00:00Z"])

    def test_forme_diverse_nella_stessa_serie(self):
        # Caso dei modelli in catena: date del CSV seguite da previsioni in ISO completo.
        self.assertEqual(
            next_timestamps(["2026-01-01", "2026-02-01", "2026-03-01T00:00:00Z"], 1),
            ["2026-04-01T00:00:00Z"],
        )

    def test_anno_e_mese(self):
        self.assertEqual(next_timestamps(["1982-01", "1982-02", "1982-03"], 1), ["1982-04-01T00:00:00Z"])


class TSDecoderForecasterTest(unittest.TestCase):
    def test_finestra_e_orizzonte(self):
        self.assertEqual(DECODER.min_points, 90)
        result = DECODER(wave(120), 7)
        self.assertEqual(len(result), 7)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_deterministico(self):
        self.assertEqual(DECODER(wave(90), 5), DECODER(wave(90), 5))

    def test_indipendente_dalla_scala(self):
        base = np.asarray(DECODER(wave(90), 5))
        scaled = np.asarray(DECODER((np.asarray(wave(90)) * 100 + 7).tolist(), 5))
        np.testing.assert_allclose(scaled, base * 100 + 7, rtol=1e-4)

    def test_dipende_dall_ingresso(self):
        self.assertFalse(np.allclose(DECODER(wave(90), 6), DECODER(wave(90, offset=3), 6)))


class ForecastErrorsTest(unittest.TestCase):
    def test_metriche_base(self):
        from model.metrics import ForecastErrors
        m = ForecastErrors([2, 4, 6], [1, 4, 8]).summary()
        self.assertAlmostEqual(m["me"], (1 + 0 - 2) / 3)
        self.assertAlmostEqual(m["mae"], 1)
        self.assertAlmostEqual(m["rmse"], (5 / 3) ** 0.5)
        self.assertAlmostEqual(m["mape"], (1 + 0 + 0.25) / 3)
        self.assertEqual(m["points"], 3)

    def test_niente_divisioni_per_zero(self):
        from model.metrics import ForecastErrors
        m = ForecastErrors([1, 1], [0, 0]).summary()
        self.assertIsNone(m["mape"])
        self.assertIsNone(m["corr"])
        self.assertAlmostEqual(m["minmax"], 1.0)  # massimo errore relativo
        self.assertAlmostEqual(m["smape"], 2.0)

    def test_mase_rispetto_al_naive(self):
        from model.metrics import ForecastErrors
        # Storico che sale di 1 a passo: il naive sbaglia di 1; qui l'errore medio è 0.5 -> MASE 0.5.
        self.assertAlmostEqual(ForecastErrors([4.5], [5], history=[1, 2, 3, 4]).mase(), 0.5)
        self.assertIsNone(ForecastErrors([1], [1]).mase())

    def test_autocorrelazione_errori(self):
        from model.metrics import ForecastErrors
        self.assertAlmostEqual(ForecastErrors([1, -1, 1, -1], [0, 0, 0, 0]).acf1(), -0.75)

    def test_compatibile_con_main(self):
        from model.metrics import forecast_accuracy
        self.assertEqual(set(forecast_accuracy([1, 2, 3], [1, 2, 4])),
                         {"mape", "me", "mae", "mpe", "rmse", "corr", "minmax", "acfe"})

    def test_lunghezze_diverse(self):
        from model.metrics import ForecastErrors
        with self.assertRaises(ValueError):
            ForecastErrors([1, 2], [1])


class AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def predict(self, **payload):
        return self.client.post("/predict", json=payload)

    def series(self, n):
        return [{"timestamp": f"2026-01-{i + 1:02d}", "value": float(i)} for i in range(n)]

    def test_status(self):
        self.assertEqual(self.client.get("/status").get_json()["models"], ["TSDecoder", "linear", "naive", "seasonal"])

    def test_elenco_modelli_con_requisiti(self):
        models = {m["name"]: m["min_points"] for m in self.client.get("/models").get_json()}
        self.assertEqual(models, {"TSDecoder": 90, "linear": 2, "naive": 1, "seasonal": 12})

    def test_previsione_lineare(self):
        body = self.predict(model="linear", time_series=self.series(3), request_id="r1",
                            auth={"username": "u"}, horizon=2).get_json()
        self.assertEqual(body["request_id"], "r1")
        self.assertEqual(body["auth"], {"username": "u"})
        self.assertEqual([p["timestamp"] for p in body["predictions"]],
                         ["2026-01-04T00:00:00Z", "2026-01-05T00:00:00Z"])
        np.testing.assert_allclose([p["value"] for p in body["predictions"]], [3, 4], atol=1e-9)

    def test_variabili_dalla_richiesta(self):
        body = self.predict(model="linear", time_series=self.series(3), env={"DEFAULT_HORIZON": "4"}).get_json()
        self.assertEqual(len(body["predictions"]), 4)
        body = self.predict(model="linear", time_series=self.series(3), env={"DEFAULT_HORIZON": "4"}, horizon=2).get_json()
        self.assertEqual(len(body["predictions"]), 2)
        response = self.predict(model="linear", time_series=self.series(3), env={"DEFAULT_HORIZON": "tanti"})
        self.assertEqual(response.status_code, 400)

    def test_endpoint_metriche(self):
        body = self.client.post("/metrics", json={"forecast": [2, 4], "actual": [1, 4], "history": [0, 1, 2]}).get_json()
        self.assertAlmostEqual(body["mae"], 0.5)
        self.assertAlmostEqual(body["mase"], 0.5)
        self.assertEqual(self.client.post("/metrics", json={"forecast": [1], "actual": []}).status_code, 400)

    def test_endpoint_federato(self):
        series = lambda scale: [{"timestamp": f"2026-01-{i + 1:02d}", "value": scale * np.sin(i / 2)} for i in range(28)]
        body = self.client.post("/federated", json={"target": series(1), "clients": [series(5)], "order": 3, "horizon": 3}).get_json()
        self.assertEqual([p["timestamp"] for p in body["predictions"]],
                         ["2026-01-29T00:00:00Z", "2026-01-30T00:00:00Z", "2026-01-31T00:00:00Z"])
        self.assertEqual(self.client.post("/federated", json={"target": []}).status_code, 400)

    def test_rifiuti_con_motivo(self):
        cases = [
            ({"model": "boh", "time_series": self.series(3)}, "Modello sconosciuto"),
            ({"model": "TSDecoder", "time_series": self.series(3)}, "almeno 90 punti"),
            ({"model": "linear", "time_series": self.series(1)}, "almeno 2 punti"),
            ({"model": "linear", "time_series": self.series(3), "horizon": 0}, "orizzonte"),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected):
                response = self.predict(**payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn(expected, response.get_json()["errore"])


if __name__ == "__main__":
    unittest.main()
