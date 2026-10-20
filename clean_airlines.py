#!/usr/bin/env python3
import argparse
import gc
import json
import re
from pathlib import Path

import pandas as pd

FILES = {
    'users': ['user_id', 'first_name', 'last_name', 'full_name'],
    'flights': ['flight_id', 'flight_date', 'flight_number', 'departure_airport', 'arrival_airport'],
    'user_flights': ['user_id', 'flight_id'],
    'user_loyalty': ['user_id', 'program_code', 'card_number'],
}
NAME_MARKERS = {'', '-', '--', 'null', 'none', 'nan', 'n/a', 'unknown', 'неизвестно'}
ALPHA = r'[^\W\d_]'


def load_table(folder, name):
    frame = pd.read_csv(folder / f'{name}.csv', sep=';', encoding='utf-8-sig',
                        dtype='string', keep_default_na=False, skip_blank_lines=False)
    missing = set(FILES[name]) - set(frame.columns)
    if missing:
        raise ValueError(f'{name}: отсутствуют обязательные столбцы {sorted(missing)}')
    return frame


def normalize(frame):
    for col in frame:
        frame[col] = (frame[col].str.normalize('NFC')
                      .str.replace(r'[\u200b\ufeff]', '', regex=True)
                      .str.replace(r'\s+', ' ', regex=True).str.strip())
    for col in ['first_name', 'middle_name', 'last_name', 'full_name']:
        if col in frame:
            frame.loc[frame[col].str.casefold().isin(NAME_MARKERS), col] = ''
    for col in ['departure_airport', 'arrival_airport', 'flight_number',
                'program_code', 'booking_code', 'travel_class', 'meal']:
        if col in frame:
            frame[col] = frame[col].str.upper()
    return frame


def save_csv(frame, path):
    frame.to_csv(path, sep=';', index=False, encoding='utf-8-sig', chunksize=100000)


def clean(input_dir, output_dir, require_split_names=False):
    src, out = Path(input_dir).resolve(), Path(output_dir).resolve()
    if src == out or src.is_relative_to(out):
        raise ValueError('Результат должен находиться в отдельной новой папке.')
    for name in FILES:
        if not (src / f'{name}.csv').is_file():
            raise FileNotFoundError(f'Не найден {src / (name + ".csv")}')
    if out.exists():
        raise FileExistsError(f'Папка {out} уже существует. Укажите новое имя, например cleaned_v2.')
    out.mkdir(parents=True)
    (out / 'rejected').mkdir()
    (out / 'quality').mkdir()
    report = {'name_policy': 'split_names_required' if require_split_names else 'split_names_or_full_name',
              'pandas_version': pd.__version__, 'tables': {}, 'checks': {}}

    def prepare(name):
        d = load_table(src, name)
        stat = {'input_rows': len(d), 'removed': {}}
        report['tables'][name] = stat
        stat['missing_before'] = {c: int(v) for c, v in d.eq('').sum().items()}
        normalize(d)

        def reject(mask, reason):
            nonlocal d
            mask = mask.fillna(False)
            count = int(mask.sum())
            stat['removed'][reason] = count
            if count:
                save_csv(d.loc[mask], out / 'rejected' / f'{name}__{reason}.csv')
                d = d.loc[~mask].copy()

        reject(d.eq('').all(axis=1), 'empty_row')
        reject(d.duplicated(), 'exact_duplicate_after_normalization')
        return lambda: d, reject, stat

    def assert_key(d, key, name):
        if d[key].eq('').any() or d[key].duplicated().any():
            save_csv(d.loc[d[key].eq('') | d[key].duplicated(keep=False)],
                     out / 'quality' / f'{name}__invalid_primary_key.csv')
            raise ValueError(f'{name}: пустой или неоднозначный {key}. См. quality; очистка остановлена.')

    def quality(d, name, keycols):
        problems = []
        def flag(mask, column, reason):
            mask = mask.fillna(False)
            if mask.any():
                part = d.loc[mask, keycols].copy()
                part['column'] = column
                part['value'] = d.loc[mask, column]
                part['issue'] = reason
                problems.append(part)
        for col in ['birth_date_iso', 'flight_date', 'arrival_date']:
            if col in d:
                parsed = pd.to_datetime(d[col], format='%Y-%m-%d', errors='coerce')
                valid_format = d[col].str.fullmatch(r'\d{4}-\d{2}-\d{2}')
                flag(d[col].ne('') & (parsed.isna() | ~valid_format), col, 'invalid_iso_date')
        for col in ['flight_time', 'arrival_time']:
            if col in d:
                flag(d[col].ne('') & ~d[col].str.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d'), col, 'invalid_time')
        for col in ['departure_airport', 'arrival_airport']:
            if col in d:
                flag(d[col].ne('') & ~d[col].str.fullmatch('[A-Z]{3}'), col, 'invalid_iata_format')
        for col in ['needs_review', 'user_needs_review', 'flight_needs_review', 'codeshare']:
            if col in d:
                flag(~d[col].str.casefold().isin(['', 'true', 'false']), col, 'invalid_boolean')
        if 'sex' in d:
            flag(~d.sex.isin(['', 'Male', 'Female']), 'sex', 'unexpected_sex_value')
        if 'observation_count' in d:
            flag(~d.observation_count.str.fullmatch(r'[1-9]\d*'), 'observation_count', 'invalid_positive_integer')
        if name == 'users':
            for col in ['first_name', 'middle_name', 'last_name', 'full_name']:
                if col in d:
                    flag(d[col].str.contains(r'\d', regex=True), col, 'name_contains_digit')
            subset = [c for c in ['first_name', 'last_name', 'birth_date_iso'] if c in d]
            eligible = d[subset].ne('').all(axis=1)
            dupe = eligible & d.duplicated(subset, keep=False)
            report['checks']['users_matching_name_and_birth_rows'] = int(dupe.sum())
            if dupe.any():
                save_csv(d.loc[dupe], out / 'quality' / 'users__matching_name_and_birth.csv')
        if name == 'flights':
            subset = ['flight_date', 'flight_number', 'departure_airport', 'arrival_airport']
            eligible = d[subset].ne('').all(axis=1)
            dupe = eligible & d.duplicated(subset, keep=False)
            report['checks']['flights_matching_route_and_date_rows'] = int(dupe.sum())
            if dupe.any():
                save_csv(d.loc[dupe], out / 'quality' / 'flights__matching_route_and_date.csv')
        report['tables'][name]['quality_issue_entries'] = sum(len(p) for p in problems)
        if problems:
            save_csv(pd.concat(problems, ignore_index=True), out / 'quality' / f'{name}__issues.csv')

    def finish(d, name, stat, keycols):
        quality(d, name, keycols)
        stat['output_rows'] = len(d)
        stat['missing_after'] = {c: int(v) for c, v in d.eq('').sum().items()}
        stat['fully_empty_columns_after'] = [c for c in d if len(d) and d[c].eq('').all()]
        assert stat['input_rows'] == len(d) + sum(stat['removed'].values())
        save_csv(d, out / f'{name}.csv')
        print(f'{name}: {stat["input_rows"]:,} -> {len(d):,}', flush=True)

    get, reject, stat = prepare('users')
    u = get()
    first = u.first_name.str.contains(ALPHA, regex=True)
    last = u.last_name.str.contains(ALPHA, regex=True)
    full = u.full_name.str.count(r'[^\W\d_]+').ge(2)
    usable = first & last
    if not require_split_names:
        usable = usable | full
    reject(~usable, 'missing_usable_name')
    u = get()
    assert_key(u, 'user_id', 'users')
    report['checks']['users_kept_using_full_name'] = int((~(first & last) & full & usable).sum())
    users = set(u.user_id)
    finish(u, 'users', stat, ['user_id'])
    del u, get, reject
    gc.collect()

    get, reject, stat = prepare('flights')
    f = get()
    assert_key(f, 'flight_id', 'flights')
    flights = set(f.flight_id)
    finish(f, 'flights', stat, ['flight_id'])
    del f, get, reject
    gc.collect()

    for name in ['user_loyalty', 'user_flights']:
        get, reject, stat = prepare(name)
        reject(~get().user_id.isin(users), 'user_not_in_clean_users')
        if name == 'user_flights':
            reject(~get().flight_id.isin(flights), 'flight_not_in_flights')
        else:
            reject(get().program_code.eq('') | get().card_number.eq(''), 'missing_loyalty_key')
        d = get()
        if name == 'user_flights':
            dupe = d.duplicated(['user_id', 'flight_id'], keep=False)
            report['checks']['user_flights_repeated_pair_rows'] = int(dupe.sum())
            if dupe.any():
                save_csv(d.loc[dupe], out / 'quality' / 'user_flights__repeated_pairs.csv')
            report['checks']['dangling_flight_references_after'] = int((~d.flight_id.isin(flights)).sum())
        report['checks'][f'{name}_dangling_user_references_after'] = int((~d.user_id.isin(users)).sum())
        finish(d, name, stat, ['user_id'] + (['flight_id'] if name == 'user_flights' else ['program_code', 'card_number']))
        del d, get, reject
        gc.collect()

    with (out / 'cleaning_report.json').open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f'Готово: {out}\nОтчёт: cleaning_report.json; удалённые строки: rejected/; проверки: quality/')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Папка с четырьмя исходными CSV, не RAR')
    parser.add_argument('--output', required=True, help='Новая папка для результата')
    parser.add_argument('--require-split-names', action='store_true',
                        help='Строго требовать first_name И last_name, даже при заполненном full_name')
    args = parser.parse_args()
    try:
        clean(args.input, args.output, args.require_split_names)
    except (ValueError, FileNotFoundError, FileExistsError, pd.errors.ParserError) as exc:
        parser.exit(1, f'Ошибка: {exc}\nЕсли появилась неполная папка результата, для повтора укажите новое имя.\n')


if __name__ == '__main__':
    main()
