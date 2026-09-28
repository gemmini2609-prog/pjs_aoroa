"""Data validation, fixed forecast evaluation, and customer segmentation.

No experiment outputs are required as runtime inputs. Forecast parameters were
selected previously; the final 28 days are evaluated without retuning.
"""
from functools import partial
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from forecast_core import (
    FINAL_FOLD, SELECTION_FOLDS, WeekdayMean56, backtest, build_daily,
    validate_targets,
)
from forecast_ridge import CalendarRidge


def load_data(path):
    """Keep valid transactions, including payment-method missingness and outliers."""
    raw = pd.read_csv(path, parse_dates=['date'])
    if raw.empty or raw[['date', 'user_id']].isna().any().any():
        raise ValueError('Missing dates or customer IDs.')
    for name in ['paid_amount', 'discount_rate', 'app_time_min']:
        raw[name] = pd.to_numeric(raw[name], errors='raise')
        if not np.isfinite(raw[name]).all():
            raise ValueError(f'Nonfinite values in {name}.')
    if (raw.paid_amount <= 0).any() or (raw.app_time_min < 0).any():
        raise ValueError('Invalid transaction amount or app usage duration.')
    if not raw.discount_rate.between(0, 1).all():
        raise ValueError('Discount rates must be between zero and one.')
    # Checks date continuity, normalized dates, and preservation of total sales.
    build_daily(raw)
    return raw


def run_forecast(raw, root, output, evaluation):
    spec = json.loads((root / 'config/model.json').read_text(encoding='utf-8'))
    factory = partial(CalendarRidge, **spec['parameters'])
    daily = build_daily(raw)
    dates = validate_targets(
        pd.read_csv(root / 'data/forecast_target_dates.csv', parse_dates=['date']).date,
        daily.index[-1],
    )
    scores, predictions = [], []
    for name, constructor in [('baseline', WeekdayMean56), (spec['model_id'], factory)]:
        for label, folds in [('selection', SELECTION_FOLDS), ('final', [FINAL_FOLD])]:
            predicted, result = backtest(constructor, daily, folds)
            scores.append(result.assign(model=name, evaluation=label))
            predictions.append(predicted.assign(model=name, evaluation=label))
    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(evaluation / 'forecast_scores.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(evaluation / 'forecast_predictions.csv', index=False)
    model = factory().fit(daily)
    prediction = model.predict(dates)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Forecast must contain finite nonnegative amounts.')
    frame = pd.DataFrame({'date': dates, 'predicted_sales': prediction.to_numpy()})
    frame.to_csv(output / 'submission_forecast.csv', index=False,
                 date_format='%Y-%m-%d', float_format='%.8f')
    model.components(dates).to_csv(evaluation / 'forecast_components.csv', index_label='date')
    print(scores.loc[scores.evaluation.eq('final'), ['model', 'mape_pct', 'mae']].to_string(index=False))
    print(f'Forecast: {len(frame)} dates, total {prediction.sum():,.2f}')
    return scores, pd.concat(predictions, ignore_index=True), frame


def customer_features(raw):
    months = pd.period_range(raw.date.min(), raw.date.max(), freq='M')
    complete = [month for month in months
                if month.start_time >= raw.date.min().normalize()
                and month.end_time.normalize() <= raw.date.max().normalize()]
    if not complete:
        raise ValueError('At least one fully observed month is required.')
    features = raw.assign(discount_used=raw.discount_rate.gt(0)).groupby('user_id').agg(
        discount_usage_fraction=('discount_used', 'mean'),
        transactions=('paid_amount', 'size'),
        total_paid=('paid_amount', 'sum'),
        mean_paid=('paid_amount', 'mean'),
        last_purchase=('date', 'max'),
    )
    monthly = pd.crosstab(raw.user_id, raw.date.dt.to_period('M')).reindex(
        index=features.index, columns=complete, fill_value=0)
    features['active_months_full'] = monthly.gt(0).sum(axis=1)
    features['recency_days'] = (raw.date.max() - features.last_purchase).dt.days
    return features


def remap_labels(features, labels):
    """Stable IDs: highest discount use first, others by descending activity."""
    profile = features.assign(label=labels).groupby('label').agg(
        discount=('discount_usage_fraction', 'median'),
        activity=('active_months_full', 'median'),
    )
    high = profile.discount.idxmax()
    order = [high] + profile.drop(index=high).sort_values('activity', ascending=False).index.tolist()
    mapping = {label: i + 1 for i, label in enumerate(order)}
    return np.array([mapping[label] for label in labels])


def run_segments(raw, output, evaluation):
    features = customer_features(raw)
    scaled = StandardScaler().fit_transform(
        features[['discount_usage_fraction', 'active_months_full']])
    scores, labels = [], {}
    for k in range(2, 6):
        labels[k] = KMeans(n_clusters=k, n_init=30, random_state=42).fit_predict(scaled)
        scores.append(dict(k=k, silhouette=silhouette_score(scaled, labels[k]),
                           davies_bouldin=davies_bouldin_score(scaled, labels[k])))
    scores = pd.DataFrame(scores)
    chosen = int(scores.loc[scores.silhouette.idxmax(), 'k'])
    features['segment'] = remap_labels(features, labels[chosen])
    submission = features[['segment']].reset_index()
    if submission.user_id.duplicated().any() or set(submission.user_id) != set(raw.user_id):
        raise ValueError('Customer coverage mismatch.')
    submission.to_csv(output / 'submission_segments.csv', index=False)
    scores.to_csv(evaluation / 'cluster_selection.csv', index=False)
    features.to_csv(evaluation / 'customer_features.csv')
    profile = features.groupby('segment').agg(
        customers=('transactions', 'size'),
        discount_usage_median=('discount_usage_fraction', 'median'),
        active_months_median=('active_months_full', 'median'),
        transactions_median=('transactions', 'median'),
        recency_days_median=('recency_days', 'median'),
        mean_paid_median=('mean_paid', 'median'),
        total_revenue=('total_paid', 'sum'),
    )
    profile.to_csv(evaluation / 'cluster_profiles.csv')
    print(f'Segments: selected k={chosen} by silhouette; {len(submission)} customers')
    print(profile.to_string())
    return features, scores, profile


def run(root, output, stage='all'):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evaluation = output / 'sourcecode/evaluation'
    evaluation.mkdir(parents=True, exist_ok=True)
    print('[1/4] Validate transactions and build features.', flush=True)
    raw = load_data(root / 'data/ecommerce_events.csv')
    q1, q3 = raw.paid_amount.quantile([.25, .75])
    upper = q3 + 1.5 * (q3 - q1)
    complete = [month for month in pd.period_range(raw.date.min(), raw.date.max(), freq='M')
                if month.start_time >= raw.date.min().normalize()
                and month.end_time.normalize() <= raw.date.max().normalize()]
    audit = dict(transactions=len(raw), customers=int(raw.user_id.nunique()),
                 days=int(raw.date.nunique()), start=str(raw.date.min().date()), end=str(raw.date.max().date()),
                 duplicate_rows=int(raw.duplicated().sum()),
                 missing={name: int(n) for name, n in raw.isna().sum().items()},
                 total_paid=float(raw.paid_amount.sum()), amount_iqr_upper=float(upper),
                 amount_iqr_above=int(raw.paid_amount.gt(upper).sum()),
                 amount_2000_rows=int(raw.paid_amount.eq(2000).sum()),
                 app_time_1_rows=int(raw.app_time_min.eq(1).sum()), complete_months=len(complete),
                 python=platform.python_version(),
                 input_sha256=hashlib.sha256((root / 'data/ecommerce_events.csv').read_bytes()).hexdigest())
    (evaluation / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if stage in ['all', 'forecast']:
        print('[2/4] Fit and evaluate sales forecasts.', flush=True)
        forecast = run_forecast(raw, root, output, evaluation)
    if stage in ['all', 'segments']:
        print('[3/4] Select cluster count and evaluate customer segments.', flush=True)
        segments = run_segments(raw, output, evaluation)
    if stage == 'all':
        print('[4/4] Generate analysis figures.', flush=True)
        from visualization import generate_figures
        generate_figures(raw, forecast, segments, output)
    print(f'Done. Output: {output}', flush=True)
