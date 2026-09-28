import unittest
import numpy as np
import pandas as pd

from forecast_core import (Fold, WeekdayMean56, build_daily, forecast_at_origin,
                           backtest, metrics, validate_targets, SELECTION_FOLDS)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=140)
        self.daily = pd.DataFrame({'sales': 100 + np.arange(140) + dates.dayofweek * 10,
                                   'transactions': np.arange(140) + 1}, index=dates)
        self.fold = Fold('synthetic', str(dates[111].date()))

    def test_exact_last_56_days(self):
        prediction = forecast_at_origin(WeekdayMean56, self.daily, self.fold)
        for date, value in prediction.items():
            dates = [d for d in self.daily.index[56:112] if d.dayofweek == date.dayofweek]
            self.assertEqual(len(dates), 8)
            self.assertAlmostEqual(value, sum(self.daily.loc[d, 'sales'] for d in dates) / 8)
        altered = self.daily.copy()
        altered.iloc[:56, 0] = 999999
        pd.testing.assert_series_equal(prediction, forecast_at_origin(WeekdayMean56, altered, self.fold))

    def test_future_changes_and_removal_do_not_change_forecast(self):
        expected = forecast_at_origin(WeekdayMean56, self.daily, self.fold)
        altered = self.daily.copy()
        altered.iloc[112:, :] = np.nan
        pd.testing.assert_series_equal(expected, forecast_at_origin(WeekdayMean56, altered, self.fold))
        pd.testing.assert_series_equal(expected, forecast_at_origin(WeekdayMean56, self.daily.iloc[:112], self.fold))

    def test_interface_only_receives_training_prefix(self):
        cutoff = pd.Timestamp(self.fold.origin)
        class Spy(WeekdayMean56):
            def fit(model, train):
                assert train.index.max() == cutoff
                assert len(train) == 112
                return super(Spy, model).fit(train)
        predictions, scores = backtest(Spy, self.daily, [self.fold])
        self.assertEqual(len(predictions), 28)
        self.assertEqual(scores.iloc[0].n_days, 28)

    def test_bad_prediction_dates_rejected(self):
        class WrongDates(WeekdayMean56):
            def predict(model, dates):
                return super(WrongDates, model).predict(dates).iloc[:-1]
        with self.assertRaises(ValueError):
            forecast_at_origin(WrongDates, self.daily, self.fold)

    def test_calendar_and_history_validation(self):
        for frame in [self.daily.iloc[:55], self.daily.drop(self.daily.index[60]),
                      self.daily.iloc[::-1], pd.concat([self.daily, self.daily.iloc[-1:]])]:
            with self.assertRaises(ValueError):
                WeekdayMean56().fit(frame)
        with self.assertRaises(ValueError):
            validate_targets(self.fold.dates.delete(3), self.fold.origin)
        with self.assertRaises(ValueError):
            validate_targets(self.fold.dates + pd.Timedelta(days=1), self.fold.origin)

    def test_aggregation_preserves_transactions(self):
        raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02']),
                            'paid_amount': [2000, 1000000, 5000]})
        daily = build_daily(raw)
        self.assertEqual(daily.sales.sum(), 1007000)
        self.assertEqual(daily.iloc[0].transactions, 2)
        self.assertEqual(daily.iloc[0].average_order_value, 501000)
        raw.loc[2, 'date'] = pd.Timestamp('2024-01-03')
        with self.assertRaises(ValueError):
            build_daily(raw)

    def test_metrics_known_values(self):
        result = metrics([100, 200], [110, 180])
        self.assertAlmostEqual(result['mape_pct'], 10)
        self.assertAlmostEqual(result['mae'], 15)
        self.assertAlmostEqual(result['total_bias_pct'], -100 / 30)
        self.assertAlmostEqual(result['total_absolute_error_pct'], 100 / 30)
        with self.assertRaises(ValueError):
            metrics([0], [10])

    def test_selection_folds_exclude_final_period(self):
        dates = [date for fold in SELECTION_FOLDS for date in fold.dates]
        self.assertEqual(len(set(dates)), 112)
        self.assertEqual(max(dates), pd.Timestamp('2024-05-05'))


if __name__ == '__main__':
    unittest.main()
