"""Generate the three analysis figures used by the separate submission report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import font_manager


ENGLISH = {1: 'Discount-using regulars', 2: 'Low-discount regulars', 3: 'Declining activity'}
COLORS = {1: '#2479AD', 2: '#E38A28', 3: '#458B65'}


def save_cluster_3d(raw, features, path):
    """Original three-axis view; mean transaction amount is display-only."""
    complete = [m for m in pd.period_range(raw.date.min(), raw.date.max(), freq='M')
                if m.start_time >= raw.date.min().normalize()
                and m.end_time.normalize() <= raw.date.max().normalize()]
    nmonths = len(complete)
    family = 'DejaVu Sans'
    korean = False
    for font in [Path('/mnt/c/Windows/Fonts/malgun.ttf'), Path('C:/Windows/Fonts/malgun.ttf')]:
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            family = font_manager.FontProperties(fname=str(font)).get_name()
            korean = True
            break
    if not korean:
        available = {font.name for font in font_manager.fontManager.ttflist}
        for name in ['Noto Sans CJK KR', 'NanumGothic', 'Malgun Gothic']:
            if name in available:
                family, korean = name, True
                break
    with plt.style.context('seaborn-v0_8-whitegrid'), plt.rc_context({
            'font.family': family, 'font.size': 12, 'axes.unicode_minus': False}):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_axes([.035, .17, .8, .71], projection='3d')
        for segment, group in features.groupby('segment'):
            label = f'군집 {segment} · {len(group)}명' if korean else f'Cluster {segment} (n={len(group)})'
            ax.scatter(group.discount_usage_fraction * 100, group.active_months_full,
                       group.mean_paid / 1000, s=23, color=COLORS.get(segment, 'gray'),
                       alpha=.65, linewidths=0, depthshade=False, label=label)
        ax.set(xlim=(0, 100), ylim=(0, nmonths),
               zlim=(0, np.ceil(features.mean_paid.max() / 10000) * 10))
        ax.set_xlabel('전체 거래의 할인 거래 비중 (%)' if korean else 'Discounted transactions (%)', labelpad=15)
        ax.set_ylabel(f'전체 {nmonths}개월 중 구매한 월 수' if korean else f'Active months out of {nmonths}', labelpad=17)
        ax.set_yticks([0, 5, 10, 15, 20, nmonths])
        ax.set_box_aspect((1.15, 1, 1))
        ax.view_init(elev=24, azim=-55)
        fig.text(.94, .53, '평균 거래액 (천 단위)' if korean else 'Mean transaction amount (thousands)',
                 rotation=90, ha='center', va='center', fontsize=12)
        k = features.segment.nunique()
        fig.suptitle(f'전체 관측 기간의 구매 활동 · {k}군집' if korean else f'Full-history customer activity: {k} clusters',
                     fontsize=19, y=.97)
        fig.legend(loc='upper center', bbox_to_anchor=(.5, .93), ncol=k, fontsize=11)
        note = (f'월 집계: {complete[0]} ~ {complete[-1]} ({nmonths}개월) · 불완전한 월은 월 집계에서 제외\n'
                '학습: 전체 할인 거래 비중 + 전체 구매 월 수 (표준화, 로그 없음)\n'
                '평균 거래액은 표시용 · 표시 축은 모두 선형 · 점 위치 이동 없음') if korean else (
                f'Complete months: {complete[0]} to {complete[-1]} ({nmonths} months)\n'
                'Training: discount usage + active months (standardized)\n'
                'Mean amount is display-only; all axes linear; no jitter')
        fig.text(.5, .075, note, ha='center', fontsize=11)
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)


def save_figures(raw, daily, features, cluster_scores, predictions, future, figures):
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes[0, 0].plot(daily.index, daily / 1e6, alpha=.4, lw=.7, label='Daily')
    axes[0, 0].plot(daily.index, daily.rolling(28).mean() / 1e6, color='#d98a28', label='Trailing 28-day mean')
    axes[0, 0].set(title='Daily sales and historical trend', ylabel='Sales (millions)')
    axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    axes[0, 0].legend()
    axes[0, 1].bar(range(7), daily.groupby(daily.index.dayofweek).mean() / 1e6, color='#2479ad')
    axes[0, 1].set(xticks=range(7), xticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                   title='Mean sales by weekday', ylabel='Sales (millions)')
    axes[1, 0].hist(raw.paid_amount, bins=np.geomspace(raw.paid_amount.min(), raw.paid_amount.max(), 45), color='#2479ad')
    axes[1, 0].set(xscale='log', title='Transaction amounts (all retained)', xlabel='Amount (log axis)', ylabel='Transactions')
    missing = raw.isna().sum()
    axes[1, 1].barh(missing.index, missing.values, color='#d98a28')
    axes[1, 1].set(title='Missing values', xlabel='Rows')
    fig.savefig(figures / '01_data.png', dpi=150)
    plt.close(fig)

    save_cluster_3d(raw, features, figures / '02_segments.png')

    final = predictions.loc[predictions.evaluation.eq('final')]
    first = final.drop_duplicates('date').sort_values('date')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained')
    axes[0].plot(first.date, first.actual_sales / 1e6, 'o-', ms=3, color='#333333', label='Actual')
    for name, group in final.groupby('model', sort=False):
        axes[0].plot(group.date, group.predicted_sales / 1e6, label=name)
    axes[0].set(title='Final 28-day fixed-origin validation', ylabel='Sales (millions)')
    axes[0].legend()
    history = daily.iloc[-56:]
    axes[1].plot(history.index, history / 1e6, color='#333333', label='Observed sales')
    axes[1].plot(future.date, future.predicted_sales / 1e6, 'o-', ms=3, color='#2479ad', label='Submission forecast')
    axes[1].axvline(daily.index[-1], color='#777777', ls='--', lw=1)
    axes[1].set(title='Full-history refit: next 28 days (point forecasts, not intervals)', ylabel='Sales (millions)')
    axes[1].legend()
    fig.savefig(figures / '03_forecast.png', dpi=150)
    plt.close(fig)



def generate_figures(raw, forecast, segments, output):
    _, predictions, future = forecast
    features, cluster_scores, _ = segments
    daily = raw.groupby('date').paid_amount.sum()
    save_figures(raw, daily, features, cluster_scores, predictions, future, output / 'images')
