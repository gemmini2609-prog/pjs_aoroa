import unittest
import numpy as np
import pandas as pd

from forecast_core import Fold, forecast_at_origin
from forecast_ridge import CalendarRidge


class RidgeTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2022-01-01', periods=728)
        t = np.arange(len(dates)) / 365.2425
        self.daily = pd.DataFrame({'sales': 1000 + 100*t + 200*np.sin(2*np.pi*t)
                                  + 100*(dates.dayofweek >= 5)}, index=dates)
        self.fold = Fold('synthetic', str(dates[699].date()))

    def test_recovers_known_calendar_signal(self):
        predicted = forecast_at_origin(lambda: CalendarRidge(alpha=0.000001), self.daily, self.fold)
        np.testing.assert_allclose(predicted, self.daily.iloc[700:].sales, atol=.001)

    def test_no_future_leakage_including_scaler_and_weights(self):
        for half_life in [None, 365]:
            factory = lambda: CalendarRidge(half_life=half_life)
            expected = forecast_at_origin(factory, self.daily, self.fold)
            changed = self.daily.copy()
            changed.iloc[700:, 0] = 1e20
            pd.testing.assert_series_equal(expected, forecast_at_origin(factory, changed, self.fold))
            pd.testing.assert_series_equal(expected, forecast_at_origin(factory, self.daily.iloc[:700], self.fold))

    def test_components_sum_and_nonnegative_postprocessing(self):
        model = CalendarRidge().fit(self.daily.iloc[:700])
        components = model.components(self.fold.dates)
        np.testing.assert_allclose(components[['level', 'trend', 'annual', 'weekday']].sum(axis=1),
                                   model.regressor.predict(model.scaler.transform(model.features(self.fold.dates))))
        model.regressor.intercept_ = -1e9
        self.assertTrue(model.predict(self.fold.dates).eq(0).all())

    def test_old_history_influences_weighted_fit(self):
        factory = lambda: CalendarRidge(half_life=365)
        expected = forecast_at_origin(factory, self.daily, self.fold)
        changed = self.daily.copy()
        changed.iloc[:200, 0] += 1000
        self.assertGreater(np.max(np.abs(expected - forecast_at_origin(factory, changed, self.fold))), 1)


if __name__ == '__main__':
    unittest.main()
