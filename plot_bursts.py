#!/usr/bin/env python3
"""Airlines: всплески поездок. Python 3.10+, pandas, numpy, plotly.

python plot_bursts.py --input cleaned_v2 --output graphs_bursts

Правило по умолчанию: 14 календарных дней, >=4 даты полётов,
ускорение >=4, >=3 города назначения. Исходные CSV не изменяются.
Ускорение = медиана интервалов вне окна / медиана интервалов внутри.
Интервалы, пересекающие границы окна, в фон НЕ входят; разрыв между
поездками до и после окна искусственно не создаётся.
Минимум 5 фоновых интервалов. Нужна история одного user_id: варианты
UUID автоматически не объединяются. Это отбор эпизодов, не классификатор шпионов.
"""
import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def read_table(path, required, optional=()):
    if not path.is_file():
        raise ValueError(f'Не найден файл: {path}')
    with path.open('rb') as stream:
        encoding = 'utf-16' if stream.read(2) in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
    with path.open(encoding=encoding, newline='') as stream:
        first = stream.readline()
        skip = int(first.strip().lower().startswith('sep='))
        header = stream.readline() if skip else first
    def norm(s):
        return re.sub(r'\s+', '_', s.strip().lstrip('\ufeff').lower())
    choices = []
    for sep in (';', ',', '\t', '|'):
        cols = next(csv.reader([header], delimiter=sep))
        names = [norm(c) for c in cols]
        choices.append((len(set(required) & set(names)), sep, cols, names))
    _, sep, cols, names = max(choices, key=lambda x: x[0])
    if len(names) != len(set(names)):
        raise ValueError(f'{path.name}: повторяющиеся названия столбцов')
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f'{path.name}: нет столбцов {sorted(missing)}; прочитаны {names}')
    selected = set(required) | set(optional)
    frame = pd.read_csv(path, sep=sep, encoding=encoding, skiprows=skip,
                        dtype='string', keep_default_na=False,
                        usecols=[c for c, n in zip(cols, names) if n in selected])
    frame.columns = [norm(c) for c in frame.columns]
    for col in frame:
        frame[col] = frame[col].str.strip()
    return frame


def prepare(folder):
    users = read_table(folder / 'users.csv', ['user_id'],
                       ['first_name', 'middle_name', 'last_name', 'full_name'])
    flights = read_table(folder / 'flights.csv', ['flight_id', 'flight_date', 'arrival_city'],
                         ['flight_number', 'departure_airport', 'arrival_airport',
                          'departure_city', 'arrival_country'])
    links = read_table(folder / 'user_flights.csv', ['user_id', 'flight_id'])
    for frame, key, label in [(users, 'user_id', 'users'), (flights, 'flight_id', 'flights')]:
        if frame[key].eq('').any() or frame[key].duplicated().any():
            raise ValueError(f'{label}: {key} должен быть непустым и уникальным')
    if not links.user_id.isin(users.user_id).all():
        raise ValueError('user_flights: есть ссылки на отсутствующих пассажиров')
    if not links.flight_id.isin(flights.flight_id).all():
        raise ValueError('user_flights: есть ссылки на отсутствующие рейсы')
    stats = {'input_link_rows': len(links), 'users_in_table': len(users),
             'duplicate_user_flight_links': int(links.duplicated().sum())}
    events = links.drop_duplicates().merge(flights, on='flight_id', validate='many_to_one')
    events['date'] = pd.to_datetime(events.flight_date, format='%Y-%m-%d', errors='coerce')
    stats['invalid_date_events_excluded'] = int(events.date.isna().sum())
    events = events.loc[events.date.notna()].copy()
    # Города нормализуются только по регистру/пробелам. Не подменяем город
    # аэропортом: несколько аэропортов могут принадлежать одному городу.
    city = events.arrival_city.str.normalize('NFKC').str.replace(r'\s+', ' ', regex=True).str.casefold()
    city = city.mask(city.isin(['', 'nan', 'none', 'null', 'unknown', 'n/a', '-']), '')
    events['city_key'] = city
    stats['events_with_missing_city'] = int(city.eq('').sum())
    names = pd.Series('', index=users.index, dtype='string')
    for col in ['first_name', 'middle_name', 'last_name']:
        if col in users:
            names = names.str.cat(users[col], sep=' ').str.strip()
    if 'full_name' in users:
        names = names.mask(names.eq(''), users.full_name)
    users['name'] = names.mask(names.eq(''), users.user_id)
    return users[['user_id', 'name']], events, stats


COLUMNS = ['user_id', 'window_start', 'window_end', 'first_flight_in_window',
           'last_flight_in_window', 'flight_dates_in_window', 'cities_in_window',
           'city_names', 'unknown_city_events', 'inside_median_days',
           'outside_median_days', 'acceleration', 'baseline_intervals',
           'baseline_intervals_before', 'baseline_intervals_after',
           'total_flight_dates', 'candidate']


def analyze_person(uid, group, args):
    daily = group.groupby('date', sort=True).city_key.agg(lambda s: set(s) - {''})
    days = daily.index.to_numpy(dtype='datetime64[D]').astype(np.int64)
    if len(days) < 2:
        return None, [], 'less_than_two_flight_dates'
    gaps = np.diff(days)
    ends = np.searchsorted(days, days + args.window_days, side='left')
    cities = daily.tolist()
    unknown_by_day = group.assign(missing=group.city_key.eq('')).groupby('date').missing.sum()
    unknown_prefix = np.r_[0, np.cumsum(unknown_by_day.reindex(daily.index).to_numpy())]
    best, best_key, qualifying = None, None, []
    for i, j in enumerate(ends):
        if j - i < 2:
            continue
        # gaps[k] connects date[k] and date[k+1]. Only wholly outside gaps.
        before = gaps[:max(0, i - 1)]
        after = gaps[j:]
        if len(before) + len(after) < args.min_baseline_gaps:
            continue
        outside = float(np.median(np.concatenate((before, after))))
        inside = float(np.median(gaps[i:j - 1]))
        ratio = outside / inside
        known = set().union(*cities[i:j])
        candidate = (j - i >= args.min_dates and ratio >= args.min_ratio
                     and len(known) >= args.min_cities)
        row = dict(zip(COLUMNS, [uid, str(np.datetime64(int(days[i]), 'D')),
            str(np.datetime64(int(days[i] + args.window_days - 1), 'D')),
            str(np.datetime64(int(days[i]), 'D')), str(np.datetime64(int(days[j - 1]), 'D')),
            int(j - i), len(known), ' | '.join(sorted(known)),
            int(unknown_prefix[j] - unknown_prefix[i]), inside, outside, ratio,
            len(before) + len(after), len(before), len(after), len(days), bool(candidate)]))
        # Один пассажир — одна точка. Сначала подходящее окно; затем число
        # дат, ускорение, города. При равенстве сохраняется раннее окно.
        key = (bool(candidate), int(j - i), ratio, len(known))
        if best_key is None or key > best_key:
            best, best_key = row, key
        if candidate:
            qualifying.append(row)
    return best, qualifying, '' if best is not None else 'insufficient_baseline_for_window'


def save_csv(frame, path):
    frame.to_csv(path, sep=';', encoding='utf-8-sig', index=False)


def build_chart(data, args, stats, target):
    fig = go.Figure()
    max_cities = max(3, int(data.cities_in_window.max())) if len(data) else 3
    for candidate, label, symbol in [(False, 'Остальные', 'circle'),
                                      (True, 'Совпали все пороги', 'diamond')]:
        part = data.loc[data.candidate.eq(candidate)] if len(data) else data
        texts = []
        for row in part.itertuples():
            e = lambda v: html.escape(str(v))
            texts.append(f'<b>{e(row.name)}</b> · {e(row.user_id)}<br>'
                         f'{row.window_start} — {row.window_end}<br>'
                         f'Дат полётов: {row.flight_dates_in_window} · городов: {row.cities_in_window}<br>'
                         f'Медиана внутри: {row.inside_median_days:g} дн.<br>'
                         f'Медиана вне: {row.outside_median_days:g} дн.<br>'
                         f'Ускорение: {row.acceleration:.2f}×<br>'
                         f'Фоновых интервалов: {row.baseline_intervals} '
                         f'(до: {row.baseline_intervals_before}, после: {row.baseline_intervals_after})<br>'
                         f'Неизвестный город: {row.unknown_city_events} записей<br>'
                         f'{e(row.city_names)}')
        fig.add_trace(go.Scattergl(x=part.acceleration.tolist(),
            y=part.flight_dates_in_window.tolist(), mode='markers', name=label,
            text=texts, hovertemplate='%{text}<extra></extra>',
            marker=dict(size=11 if candidate else 6, symbol=symbol,
                        opacity=.95 if candidate else .35,
                        color=part.cities_in_window.tolist(), coloraxis='coloraxis',
                        line=dict(width=1.5 if candidate else 0, color='#111827'))))
    fig.add_vline(x=args.min_ratio, line_dash='dash', line_color='#64748b')
    fig.add_hline(y=args.min_dates, line_dash='dash', line_color='#64748b')
    if not len(data):
        fig.add_annotation(text='Недостаточно данных для сравнения', x=.5, y=.5,
                           xref='paper', yref='paper', showarrow=False)
    fig.update_layout(template='plotly_white', height=680,
        font=dict(family='Arial, sans-serif', size=14, color='#1e293b'),
        margin=dict(l=85, r=100, t=30, b=90),
        xaxis=dict(title='Ускорение ритма, × (медиана вне окна / медиана внутри)', type='log'),
        yaxis=dict(title=f'Различных дат полётов за {args.window_days} дней', rangemode='tozero', dtick=1),
        legend=dict(orientation='h', y=-.18, x=0),
        coloraxis=dict(cmin=0, cmax=max_cities, colorscale='Viridis',
                       colorbar=dict(title='Городов', dtick=1)))
    div = fig.to_html(full_html=False, include_plotlyjs=True,
                      config={'responsive': True, 'displaylogo': False,
                              'toImageButtonOptions': {'filename': 'flight_bursts', 'scale': 2}})
    title = 'Всплески поездок'
    subtitle = (f'{args.window_days} дней · ≥{args.min_dates} даты полётов · '
                f'ускорение ≥{args.min_ratio:g}× · ≥{args.min_cities} города')
    counts = (f'Показано: {len(data):,} · Совпали пороги: {stats["candidate_users"]:,} · '
              f'Недостаточно истории: {stats["not_assessable_users"]:,}').replace(',', ' ')
    page = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>body{{margin:0;background:#f4f6f9;color:#1e293b;'
            'font-family:Arial,sans-serif}}main{max-width:1400px;margin:24px auto;padding:24px;'
            'background:white;border-radius:16px}h1{font-size:26px;margin:0 0 12px}'
            '.meta{font-size:14px;color:#475569;margin:8px 0}@media(max-width:700px)'
            '{main{margin:0;padding:12px;border-radius:0}}</style></head><body><main>'
            f'<h1>{title}</h1><div class="meta">{subtitle}</div>'
            f'<div class="meta">{counts}</div>'
            '<div class="meta">1 точка — 1 пассажир · ромб с обводкой — совпали пороги · '
            'цвет — города · пунктир — пороги по осям · ось X логарифмическая</div>'
            f'{div}</main></body></html>')
    target.write_text(page, encoding='utf-8')


def run(args):
    source, target = args.input.resolve(), args.output.resolve()
    if not source.is_dir():
        raise ValueError(f'Нет входной папки: {source}')
    if target == source or source in target.parents:
        raise ValueError('Выходная папка должна находиться отдельно от входной')
    if target.exists() and any(target.iterdir()):
        raise ValueError(f'Выходная папка не пуста: {target}. Укажите новое имя.')
    print('Читаю таблицы…', flush=True)
    users, events, stats = prepare(source)
    rows, windows, skipped = [], [], []
    processed = set()
    print('Проверяю 14-дневные окна…' if args.window_days == 14 else 'Проверяю окна…', flush=True)
    for number, (uid, group) in enumerate(events.groupby('user_id', sort=False), 1):
        processed.add(uid)
        row, matches, reason = analyze_person(uid, group, args)
        if row is not None:
            rows.append(row)
            windows.extend(matches)
        else:
            skipped.append({'user_id': uid, 'reason': reason})
        if number % 10000 == 0:
            print(f'Обработано пассажиров: {number:,}', flush=True)
    for uid in users.user_id:
        if uid not in processed:
            skipped.append({'user_id': uid, 'reason': 'no_valid_dated_flights'})
    data = pd.DataFrame(rows, columns=COLUMNS).merge(users, on='user_id', validate='one_to_one')
    data = data.sort_values(['candidate', 'acceleration', 'flight_dates_in_window'], ascending=False)
    all_windows = pd.DataFrame(windows, columns=COLUMNS).merge(users, on='user_id', validate='many_to_one')
    excluded = pd.DataFrame(skipped, columns=['user_id', 'reason']).merge(users, on='user_id')
    candidates = data.loc[data.candidate.eq(True)].copy()
    stats.update(assessed_users=len(data), candidate_users=len(candidates),
                 qualifying_windows=len(all_windows), not_assessable_users=len(excluded),
                 parameters={k: v for k, v in vars(args).items() if k not in ['input', 'output']},
                 window_definition='[start_date, start_date + window_days); calendar dates',
                 baseline_definition='Only consecutive-date gaps wholly before or wholly after window',
                 selection='Any qualifying window first; then date count, acceleration, city count, earliest',
                 cities_definition='Distinct normalized arrival_city; unknown values excluded; no airport fallback',
                 interpretation='Rule matches, not proof of espionage; user_id identities are not resolved')
    target.mkdir(parents=True, exist_ok=True)
    save_csv(data, target / 'all_passengers.csv')
    save_csv(candidates, target / 'candidates.csv')
    save_csv(all_windows, target / 'candidate_windows.csv')
    save_csv(excluded, target / 'not_assessable.csv')
    save_csv(events.loc[events.user_id.isin(candidates.user_id)].drop(columns=['city_key']),
             target / 'candidate_flights.csv')
    (target / 'burst_report.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    build_chart(data, args, stats, target / '04_bursts.html')
    print(f'Готово: {target / "04_bursts.html"}')
    print(f'Оценено: {len(data):,}; совпали пороги: {len(candidates):,}; недостаточно истории: {len(excluded):,}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-days', type=int, default=14)
    parser.add_argument('--min-dates', type=int, default=4)
    parser.add_argument('--min-ratio', type=float, default=4)
    parser.add_argument('--min-cities', type=int, default=3)
    parser.add_argument('--min-baseline-gaps', type=int, default=5)
    args = parser.parse_args()
    if (args.window_days < 2 or not 2 <= args.min_dates <= args.window_days or
            not np.isfinite(args.min_ratio) or args.min_ratio <= 0 or
            args.min_cities < 1 or args.min_baseline_gaps < 1):
        parser.error('Недопустимые значения порогов')
    try:
        run(args)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        sys.exit(f'Ошибка: {exc}')


if __name__ == '__main__':
    main()
