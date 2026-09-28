"""Calendar-only Ridge forecasts; all fitted quantities use the training prefix."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from forecast_core import validate_daily, validate_targets


class CalendarRidge:
    def __init__(self, fourier_pairs=1, alpha=1.0, half_life=None, trend=True):
        if fourier_pairs not in range(4) or alpha <= 0 or (half_life is not None and half_life <= 0):
            raise ValueError('Invalid Ridge configuration.')
        self.fourier_pairs, self.alpha = fourier_pairs, alpha
        self.half_life, self.trend = half_life, trend

    def features(self, dates):
        dates = pd.DatetimeIndex(dates)
        elapsed = (dates - self.start).days.to_numpy() / 365.2425
        features = {f'weekday_{i}': (dates.dayofweek == i).astype(float) for i in range(7)}
        if self.trend:
            features['trend'] = elapsed
        for k in range(1, self.fourier_pairs + 1):
            features[f'annual_sin_{k}'] = np.sin(2 * np.pi * k * elapsed)
            features[f'annual_cos_{k}'] = np.cos(2 * np.pi * k * elapsed)
        return pd.DataFrame(features, index=dates)

    def fit(self, train):
        validate_daily(train)
        self.start, self.origin = train.index[0], train.index[-1]
        age = (self.origin - train.index).days.to_numpy()
        weights = np.ones(len(train)) if self.half_life is None else 2.0 ** (-age / self.half_life)
        weights /= weights.mean()  # Keep the regularization scale comparable.
        x = self.features(train.index)
        self.scaler = StandardScaler().fit(x, sample_weight=weights)
        self.regressor = Ridge(alpha=self.alpha, solver='cholesky').fit(
            self.scaler.transform(x), train.sales.to_numpy(), sample_weight=weights)
        return self

    def components(self, dates):
        dates = validate_targets(dates, self.origin)
        x = self.features(dates)
        effects = pd.DataFrame(self.scaler.transform(x) * self.regressor.coef_,
                               index=dates, columns=x.columns)
        result = pd.DataFrame(index=dates)
        result['level'] = self.regressor.intercept_
        for group in ['weekday', 'trend', 'annual']:
            result[group] = effects.filter(like=group).sum(axis=1)
        result['raw_prediction'] = result.sum(axis=1)
        # This same nonnegative postprocessing is used in validation and deployment.
        result['predicted_sales'] = result.raw_prediction.clip(lower=0)
        return result

    def predict(self, dates):
        return self.components(dates).predicted_sales
