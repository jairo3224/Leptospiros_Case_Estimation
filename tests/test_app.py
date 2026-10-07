import unittest

import pandas as pd

from app import (
    explain_replay_difference,
    get_saved_prediction,
    get_weather_week,
    target_week,
)


class ReplayHelpersTests(unittest.TestCase):
    def test_target_week_adds_selected_horizon(self) -> None:
        anchor = pd.Timestamp("2022-12-05")

        self.assertEqual(target_week(anchor, 0), anchor)
        self.assertEqual(target_week(anchor, 2), pd.Timestamp("2022-12-19"))

    def test_target_week_rejects_unsupported_horizon(self) -> None:
        with self.assertRaises(ValueError):
            target_week(pd.Timestamp("2022-12-05"), 4)

    def test_saved_prediction_matches_city_date_horizon_and_model(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2022-12-05"),
                    "city": "Dagupan City",
                    "horizon": 2,
                    "model": "XGB_full_pois",
                    "y_true": 3,
                    "y_pred": 2.5,
                },
                {
                    "date": pd.Timestamp("2022-12-05"),
                    "city": "Dagupan City",
                    "horizon": 2,
                    "model": "A_all_zero",
                    "y_true": 3,
                    "y_pred": 0,
                },
            ]
        )

        result = get_saved_prediction(
            predictions,
            "Dagupan City",
            pd.Timestamp("2022-12-05"),
            2,
            "XGB_full_pois",
        )

        self.assertEqual(result["y_true"], 3)
        self.assertEqual(result["y_pred"], 2.5)

    def test_saved_prediction_requires_exactly_one_match(self) -> None:
        predictions = pd.DataFrame(
            columns=["date", "city", "horizon", "model", "y_true", "y_pred"]
        )

        with self.assertRaises(ValueError):
            get_saved_prediction(
                predictions,
                "Dagupan City",
                pd.Timestamp("2022-12-05"),
                2,
                "XGB_full_pois",
            )

    def test_weather_week_includes_seven_days_from_selected_week_start(self) -> None:
        dates = pd.date_range("2022-12-12", periods=8, freq="D")
        weather = pd.DataFrame(
            {
                "date": dates,
                "city": ["Davao City"] * 8,
                "rainfall_mm": range(8),
            }
        )

        result = get_weather_week(
            weather,
            "Davao City",
            pd.Timestamp("2022-12-12"),
        )

        self.assertEqual(len(result), 7)
        self.assertEqual(result["date"].min(), pd.Timestamp("2022-12-12"))
        self.assertEqual(result["date"].max(), pd.Timestamp("2022-12-18"))

    def test_zero_baseline_explanation_describes_its_behavior(self) -> None:
        explanation = explain_replay_difference("A_all_zero", 0.0, 3.0)

        self.assertIn("3.00 cases lower", explanation)
        self.assertIn("always predicts zero", explanation)
        self.assertIn("one historical test week", explanation)

    def test_model_explanation_does_not_claim_a_specific_cause(self) -> None:
        explanation = explain_replay_difference("XGB_full_pois", 1.0, 2.0)

        self.assertIn("lagged weather summaries", explanation)
        self.assertIn("cannot identify which factor caused", explanation)


if __name__ == "__main__":
    unittest.main()
