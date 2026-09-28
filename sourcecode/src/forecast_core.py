"""Daily forecasting contract and fixed-origin evaluation shared by all models."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    origin: str
    purpose: str = 'selection'

    @property
    def dates(self):
        return pd.date_range(pd.Timestamp(self.origin) + pd.Timedelta(days=1), periods=28)


SELECTION_FOLDS = tuple(Fold(f'selection_{i}', origin) for i, origin in enumerate(
    ['2024-01-14', '2024-02-11', '2024-03-10', '2024-04-07'], 1))
FINAL_FOLD = Fold('final', '2024-05-05', 'reserved_M6')
SUBMISSION_FOLD = Fold('submission', '2024-06-02', 'reserved_M7')


def validate_dates(dates):
    dates = pd.DatetimeIndex(dates)
    if (len(dates) == 0 or dates.hasnans or dates.tz is not None
            or not dates.equals(dates.normalize())
            or not dates.equals(pd.date_range(dates[0], periods=len(dates)))):
        raise ValueError('Dates must be unique, ordered, consecutive, naive calendar days.')
    return dates


def validate_daily(daily):
    validate_dates(daily.index)
    values = daily['sales'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Daily sales must be finite and positive; explicit handling required.')


def build_daily(raw):
    if raw.empty or raw['date'].isna().any():
        raise ValueError('Missing transaction dates.')
    if not raw['date'].equals(raw['date'].dt.normalize()):
        raise ValueError('Expected calendar dates without times.')
    amounts = raw['paid_amount'].to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts <= 0).any():
        raise ValueError('Invalid transaction amounts.')
    # Multiple transactions per day are expected; no rows are removed.
    daily = raw.groupby('date', sort=True).agg(
        sales=('paid_amount', 'sum'), transactions=('paid_amount', 'size'))
    daily['average_order_value'] = daily.sales / daily.transactions
    validate_daily(daily)  # A missing day is not silently replaced by zero.
    if not np.isclose(daily.sales.sum(), raw.paid_amount.sum(), rtol=1e-12, atol=1e-6):
        raise ValueError('Raw and aggregated totals differ.')
    return daily


def validate_targets(dates, origin):
    dates = validate_dates(dates)
    expected = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=28)
    if not dates.equals(expected):
        raise ValueError('Targets must be the 28 days immediately after the origin.')
    return dates


class ForecastModel(Protocol):
    def fit(self, train: pd.DataFrame): ...
    def predict(self, dates: pd.DatetimeIndex) -> pd.Series: ...


class WeekdayMean56:
    """Official baseline: unweighted weekday means over the last 56 calendar days."""
    def fit(self, train):
        validate_daily(train)
        if len(train) < 56:
            raise ValueError('At least 56 consecutive training days required.')
        self.origin = train.index[-1]
        recent = train.loc[self.origin - pd.Timedelta(days=55):, 'sales']
        self.weekday_means = recent.groupby(recent.index.dayofweek).mean()
        self.weekday_counts = recent.groupby(recent.index.dayofweek).size()
        if not self.weekday_counts.eq(8).all():
            raise ValueError('Expected eight observations per weekday.')
        return self

    def predict(self, dates):
        if not hasattr(self, 'origin'):
            raise ValueError('Fit the model first.')
        dates = validate_targets(dates, self.origin)
        return pd.Series(self.weekday_means.loc[dates.dayofweek].to_numpy(),
                         index=dates, name='predicted_sales')


def forecast_at_origin(factory, daily, fold):
    # Only this prefix is passed to the model; future observations are unnecessary.
    origin = pd.Timestamp(fold.origin)
    train = daily.loc[daily.index <= origin].copy(deep=True)
    validate_daily(train)
    if train.index[-1] != origin:
        raise ValueError('Training data does not reach the forecast origin.')
    model: ForecastModel = factory()
    model.fit(train)
    predicted = model.predict(fold.dates)
    if not isinstance(predicted, pd.Series) or not predicted.index.equals(fold.dates):
        raise ValueError('Model must return a Series indexed by the exact target dates.')
    if not np.isfinite(predicted.to_numpy()).all() or (predicted < 0).any():
        raise ValueError('Predictions must be finite and nonnegative.')
    return predicted.rename('predicted_sales')


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if (actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or (actual <= 0).any() or (predicted < 0).any()):
        raise ValueError('Metrics require aligned finite vectors and positive actual sales.')
    error = predicted - actual
    bias = 100 * error.sum() / actual.sum()
    return dict(n_days=len(actual), mape_pct=100 * np.mean(np.abs(error) / actual),
                mae=np.mean(np.abs(error)), total_bias_pct=bias,
                total_absolute_error_pct=abs(bias))


def backtest(factory, daily, folds=SELECTION_FOLDS):
    predictions, scores = [], []
    for fold in folds:
        prediction = forecast_at_origin(factory, daily, fold)
        # Actuals are fetched only after the entire 28-day forecast is generated.
        actual = daily['sales'].reindex(fold.dates)
        score = metrics(actual, prediction)
        scores.append(dict(fold=fold.name, origin=fold.origin, **score))
        frame = pd.DataFrame({'date': fold.dates, 'actual_sales': actual.to_numpy(),
                              'predicted_sales': prediction.to_numpy()})
        frame['fold'], frame['origin'] = fold.name, fold.origin
        frame['horizon'] = np.arange(1, 29)
        frame['horizon_week'] = (frame.horizon - 1) // 7 + 1
        frame['weekday'] = frame.date.dt.dayofweek
        frame['day_type'] = np.where(frame.weekday >= 5, 'weekend', 'weekday')
        predictions.append(frame)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(scores)
