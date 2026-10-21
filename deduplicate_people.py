#!/usr/bin/env python3
"""Второй этап очистки Airlines. Python 3.10+, pandas 2.2–2.x.
python deduplicate_people.py --input cleaned --output cleaned_v2
ФИО сами по себе не являются основанием объединения. Исходники не меняются.
"""
import argparse
import csv
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
import gc
import json
import re
import shutil
import tempfile
import unicodedata
import pandas as pd

RU = 'А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я'.split()
LAT = 'A B V G D E E ZH Z I I K L M N O P R S T U F KH TS CH SH SHCH - Y - E IU IA'.split()
TRANSLATION = str.maketrans(dict(zip(RU, [x if x != '-' else '' for x in LAT])))
MISSING = {'', '-', 'NONE', 'NULL', 'NAN', 'UNKNOWN', 'N/A'}


def normalize_name(value):
    value = str(value).upper().translate(TRANSLATION).replace('Y', 'I')
    value = ''.join(c for c in unicodedata.normalize('NFKD', value) if not unicodedata.combining(c))
    value = re.sub(r'[^A-Z]', '', value)
    # Распространённые варианты русской транслитерации: MAXIM/MAKSIM,
    # RUMIANTCEV/RUMIANTSEV, SCHERBAKOV/SHCHERBAKOV.
    return value.replace('X', 'KS').replace('TC', 'TS').replace('SCH', 'SHCH')


def normalize_document(value):
    text = str(value).strip().upper()
    if text in MISSING:
        return ''
    return re.sub(r'[^\w]', '', text, flags=re.UNICODE)


def one_edit(a, b):
    """Равенство либо одна вставка, удаление, замена буквы. Не произвольный fuzzy score."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 3:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def aliases(row):
    if row['first_name'] and row['last_name']:
        parts = [(normalize_name(row['first_name']), normalize_name(row['last_name']))]
    else:
        words = [normalize_name(w) for w in row['full_name'].split()]
        words = [w for w in words if len(w) >= 2]
        # Порядок имени/фамилии в full_name не угадывается; это только ключи кандидатов.
        parts = list(combinations(words, 2)) if 2 <= len(words) <= 4 else []
    return {tuple(sorted(pair)) for pair in parts if min(map(len, pair)) >= 2}


def name_match(a, b):
    if a & b:
        return True
    for x, y in product(a, b):
        if ((one_edit(x[0], y[0]) and one_edit(x[1], y[1])) or
                (one_edit(x[0], y[1]) and one_edit(x[1], y[0]))):
            return True
    return False


def history_support(shared_dates, overlap, jaccard, exact_normalized_name):
    # Либо почти одинаковые полные истории, либо >=5 общих дат и практически
    # вся меньшая история — подмножество большей при совпавшем нормализованном ФИО.
    return ((shared_dates >= 3 and overlap >= .9 and jaccard >= .7) or
            (exact_normalized_name and shared_dates >= 5 and overlap >= .95))


REQUIRED_COLUMNS = {
    'users': {'user_id', 'first_name', 'last_name', 'full_name', 'passport_key'},
    'flights': {'flight_id', 'flight_date', 'flight_number', 'departure_airport', 'arrival_airport'},
    'user_flights': {'user_id', 'flight_id'},
    'user_loyalty': {'user_id', 'program_code', 'card_number'},
}


def column_name(value):
    return re.sub(r'\s+', '_', str(value).replace('\ufeff', '').strip().lower())


def csv_format(folder, name):
    path = Path(folder) / (name + '.csv')
    with path.open('rb') as stream:
        bom = stream.read(4)
    encoding = 'utf-16' if bom.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    with path.open('r', encoding=encoding, newline='') as stream:
        first = stream.readline()
        skiprows = 0
        forced = None
        if first.strip().lower().startswith('sep=') and len(first.strip()) == 5:
            forced = first.strip()[-1]
            skiprows = 1
            first = stream.readline()
    if not first.strip():
        raise ValueError(f'{path.name}: файл пуст или отсутствует заголовок')
    candidates = []
    for delimiter in ([forced] if forced else [';', ',', '\t', '|']):
        header = next(csv.reader([first], delimiter=delimiter))
        columns = [column_name(c) for c in header]
        candidates.append((delimiter, columns))
    required = REQUIRED_COLUMNS.get(name, set())
    suitable = [(sep, cols) for sep, cols in candidates if len(cols) > 1 and required <= set(cols)]
    if suitable:
        delimiter, columns = max(suitable, key=lambda item: len(item[1]))
    else:
        delimiter, columns = max(candidates, key=lambda item: len(item[1]))
    if len(set(columns)) != len(columns):
        raise ValueError(f'{path.name}: повторяющиеся заголовки после нормализации: {columns!r}')
    missing = required - set(columns)
    if missing:
        raise ValueError(f'{path.name}: не найдены столбцы {sorted(missing)}. '
                         f'Прочитанные заголовки: {columns!r}; разделитель: {delimiter!r}. '
                         'Проверьте, что это нужная таблица, а не файл из audit.')
    return path, delimiter, encoding, skiprows, columns


def read(folder, name):
    path, delimiter, encoding, skiprows, columns = csv_format(folder, name)
    frame = pd.read_csv(path, sep=delimiter, dtype='string', keep_default_na=False,
                        encoding=encoding, skiprows=skiprows)
    frame.columns = [column_name(c) for c in frame.columns]
    if name in REQUIRED_COLUMNS:
        print(f'  {path.name}: {len(frame):,} строк; разделитель {delimiter!r}; заголовки проверены', flush=True)
    return frame


def save(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    frame.to_csv(temporary, sep=';', index=False, encoding='utf-8-sig', chunksize=100000)
    temporary.replace(path)


def _run_pass(src, out, strong_only=False):
    src, out = Path(src).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Выходная папка уже существует. Укажите новую, например cleaned_v3.')
    for name in ['users', 'flights', 'user_flights', 'user_loyalty']:
        if not (src / (name + '.csv')).is_file():
            raise ValueError(f'Не найден {name}.csv в {src}')
    u, f, x, loyalty = [read(src, name) for name in ['users', 'flights', 'user_flights', 'user_loyalty']]
    prior_report = src / 'deduplication_report.json'
    if prior_report.exists():
        expected = json.loads(prior_report.read_text(encoding='utf-8'))['output']
        actual = dict(zip(['users', 'flights', 'user_flights', 'user_loyalty'], map(len, [u, f, x, loyalty])))
        if actual != expected:
            raise ValueError('Число строк входных CSV не совпадает с предыдущим отчётом: файлы изменены или записаны не полностью')
    for frame, key in [(u, 'user_id'), (f, 'flight_id')]:
        if frame[key].eq('').any() or frame[key].duplicated().any():
            raise ValueError(f'{key}: пустые или повторяющиеся первичные ключи')
    if not x.user_id.isin(u.user_id).all() or not loyalty.user_id.isin(u.user_id).all():
        raise ValueError('Есть ссылки на отсутствующих пользователей')
    if not x.flight_id.isin(f.flight_id).all():
        raise ValueError('Есть ссылки на отсутствующие рейсы')
    out.mkdir(parents=True)
    audit = out / 'audit'
    audit.mkdir()
    report = {'strong_only': strong_only, 'input': {'users': len(u), 'flights': len(f),
               'user_flights': len(x), 'user_loyalty': len(loyalty)}}
    # Убираем только полностью одинаковые сведения рейса, исключая ID.
    fcols = [c for c in f if c != 'flight_id']
    duplicated = f.duplicated(fcols, keep=False)
    fd = f.loc[duplicated]
    fa = pd.DataFrame(columns=['old_flight_id', 'canonical_flight_id'])
    if len(fd):
        canonical = fd.groupby(fcols, dropna=False, sort=False).flight_id.transform('min')
        fa = pd.DataFrame({'old_flight_id': fd.flight_id, 'canonical_flight_id': canonical})
        fa = fa[fa.old_flight_id.ne(fa.canonical_flight_id)]
        fm = fa.set_index('old_flight_id').canonical_flight_id
        x['flight_id'] = x.flight_id.map(fm).fillna(x.flight_id)
        f = f[~f.flight_id.isin(fa.old_flight_id)].copy()
    save(fa, audit / 'flight_id_map.csv')
    report['exact_flight_ids_merged'] = len(fa)
    del fd
    gc.collect()

    n = len(u)
    rows = u.to_dict('records')
    ids = u.user_id.tolist()
    index = {value: i for i, value in enumerate(ids)}
    # При повторной проверке восстанавливаем исходные варианты из журнала.
    # Иначе выбор одного написания мог бы скрыть прежнее противоречие.
    archived_profiles = pd.DataFrame(columns=u.columns)
    previous_id_map = pd.DataFrame()
    records_per_user = [[r] for r in rows]
    archive_path = src / 'audit' / 'original_merged_user_profiles.csv'
    map_path = src / 'audit' / 'user_id_map.csv'
    if archive_path.exists() and map_path.exists():
        archived_profiles = read(src / 'audit', 'original_merged_user_profiles')
        previous_id_map = read(src / 'audit', 'user_id_map')
        previous_mapping = previous_id_map.set_index('old_user_id').canonical_user_id.to_dict()
        for record in archived_profiles.to_dict('records'):
            target = previous_mapping.get(record['user_id'], record['user_id'])
            if target in index:
                records_per_user[index[target]].append(record)
    report['archived_profiles_used_for_consistency'] = len(archived_profiles)
    name_variants = [[aliases(record) for record in variants] for variants in records_per_user]
    names = [set().union(*variants) for variants in name_variants]
    middle_variants = [{normalize_name(r.get('middle_name', '')) for r in variants
                        if normalize_name(r.get('middle_name', ''))} for variants in records_per_user]
    passport = [normalize_document(r.get('passport_key', '')) for r in rows]
    birth = [r.get('birth_date_iso', '').strip() for r in rows]
    sex = [r.get('sex', '').strip().upper() for r in rows]
    middle = [normalize_name(r.get('middle_name', '')) for r in rows]
    parent = list(range(n))
    members = {i: [i] for i in range(n)}
    component_passports = [{v} if v else set() for v in passport]
    component_births = [{v} if v else set() for v in birth]
    component_sexes = [{v} if v and v not in MISSING else set() for v in sex]
    accepted = []
    blocked = []
    reasons = Counter()

    def find(a):
        while a != parent[a]:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def compatible(a, b):
        ra, rb = find(a), find(b)
        if len(component_passports[ra] | component_passports[rb]) > 1:
            return 'different_passports'
        if len(component_births[ra] | component_births[rb]) > 1:
            return 'different_birth_dates'
        if len(component_sexes[ra] | component_sexes[rb]) > 1:
            return 'different_sex_values'
        # Проверяем все записи двух групп, а не только одну пару: защита от цепочек.
        for i, j in product(members[ra], members[rb]):
            if any(not name_match(a, b) for a, b in product(name_variants[i], name_variants[j])):
                return 'incompatible_names_in_component'
            for ma, mb in product(middle_variants[i], middle_variants[j]):
                if len(ma) == 1 or len(mb) == 1:
                    ok = ma[0] == mb[0]
                else:
                    ok = one_edit(ma, mb)
                if not ok:
                    return 'different_middle_names'
        return ''

    def merge(a, b, reason, evidence=''):
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        issue = compatible(a, b)
        if issue:
            blocked.append((ids[a], ids[b], reason, issue))
            return False
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra].extend(members.pop(rb))
        component_passports[ra] |= component_passports[rb]
        component_births[ra] |= component_births[rb]
        component_sexes[ra] |= component_sexes[rb]
        accepted.append((ids[a], ids[b], reason, evidence))
        reasons[reason] += 1
        return True

    strong_edges = defaultdict(list)

    def try_groups(groups, reason):
        for group in groups:
            if len(group) < 2:
                continue
            for a, b in combinations(sorted(set(group)), 2):
                # Номер документа или карты не разрешает игнорировать конфликт имён.
                if name_match(names[a], names[b]):
                    strong_edges[reason].append((a, b))
                elif reason == 'same_passport':
                    blocked.append((ids[a], ids[b], reason, 'different_names'))

    print('1/5 · Документы, даты рождения и карты лояльности', flush=True)
    passport_groups = defaultdict(list)
    dob_groups = defaultdict(list)
    for i in range(n):
        if passport[i]:
            passport_groups[passport[i]].append(i)
        if birth[i]:
            dob_groups[birth[i]].append(i)
    repeated_passport_rows = u[[len(passport_groups[v]) > 1 if v else False for v in passport]]
    save(repeated_passport_rows, audit / 'input_shared_passports.csv')
    try_groups(passport_groups.values(), 'same_passport')
    try_groups(dob_groups.values(), 'name_and_birth_date')
    for _, group in loyalty.groupby(['program_code', 'card_number'], sort=False):
        if group.program_code.iloc[0] and group.card_number.iloc[0] and len(group) > 1:
            try_groups([[index[v] for v in group.user_id.unique()]], 'name_and_loyalty_card')

    # Обрабатываем подтверждения пакетами, чтобы профиль без паспорта не
    # присоединился случайно к первому из двух противоречащих друг другу людей.
    for rule in ['same_passport', 'name_and_birth_date', 'name_and_loyalty_card']:
        edges = []
        neighbours = defaultdict(set)
        for a, b in strong_edges[rule]:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            issue = compatible(a, b)
            if issue:
                blocked.append((ids[a], ids[b], rule, issue))
                continue
            edges.append((a, b))
            neighbours[ra].add(rb)
            neighbours[rb].add(ra)
        ambiguous_targets = set()
        for root, others in neighbours.items():
            if any(compatible(a, b) for a, b in combinations(sorted(others), 2)):
                ambiguous_targets.add(root)
        for a, b in edges:
            if find(a) in ambiguous_targets or find(b) in ambiguous_targets:
                blocked.append((ids[a], ids[b], rule, 'ambiguous_identity_targets'))
            else:
                merge(a, b, rule)
    del strong_edges
    print('2/5 · Варианты ФИО и подтверждение историями поездок', flush=True)
    name_groups = defaultdict(list)
    for i, variants in enumerate(names):
        for key in variants:
            name_groups[key].append(i)
    candidate_pairs = set()
    def candidate(a, b):
        if a == b or find(a) == find(b):
            return
        if a > b:
            a, b = b, a
        candidate_pairs.add(a * n + b)
    for group in name_groups.values():
        for a, b in combinations(group, 2):
            candidate(a, b)
    # Для опечаток блокируем по одному совпавшему слову; не сравниваем всех со всеми.
    blocks = defaultdict(dict)
    for key in name_groups:
        a, b = key
        blocks[a][b] = key
        blocks[b][a] = key
    for anchor, words in blocks.items():
        if len(words) < 2:
            continue
        deletions = defaultdict(list)
        word_pairs = set()
        for word in words:
            if len(word) < 4:
                continue
            for short in {word[:k] + word[k+1:] for k in range(len(word))}:
                if short in words:
                    word_pairs.add(tuple(sorted((word, short))))
                deletions[short].append(word)
        for variants in deletions.values():
            for a, b in combinations(set(variants), 2):
                if one_edit(a, b):
                    word_pairs.add(tuple(sorted((a, b))))
        for a, b in word_pairs:
            for i, j in product(name_groups[words[a]], name_groups[words[b]]):
                candidate(i, j)
    report['name_candidate_pairs_after_strong_rules'] = len(candidate_pairs)
    fcode = {v: i for i, v in enumerate(f.flight_id)}
    dates_by_code = f.flight_date.tolist()
    history = {r: set() for r in members}
    for i, flight in zip(x.user_id.map(index), x.flight_id.map(fcode)):
        history[find(int(i))].add(int(flight))
    profile_edges = []
    review_pairs = []
    for key in sorted(candidate_pairs):
        a, b = divmod(key, n)
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        issue = compatible(a, b)
        if issue:
            review_pairs.append((ids[a], ids[b], issue, 0, 0.0, 0.0))
            continue
        ha, hb = history[ra], history[rb]
        shared = ha & hb
        shared_dates = len({dates_by_code[i] for i in shared if dates_by_code[i]})
        overlap = len(shared) / min(len(ha), len(hb)) if ha and hb else 0.0
        jaccard = len(shared) / len(ha | hb) if ha or hb else 0.0
        if history_support(shared_dates, overlap, jaccard, bool(names[a] & names[b])):
            profile_edges.append((a, b, shared_dates, overlap, jaccard))
        else:
            review_pairs.append((ids[a], ids[b], 'insufficient_independent_evidence',
                                 shared_dates, round(overlap, 4), round(jaccard, 4)))
    # Если безымянный в смысле документов профиль подходит двум противоречащим
    # друг другу людям, не выбираем произвольно первого.
    neighbours = defaultdict(set)
    for a, b, *_ in profile_edges:
        ra, rb = find(a), find(b)
        neighbours[ra].add(rb)
        neighbours[rb].add(ra)
    ambiguous = set()
    for root, others in neighbours.items():
        if any(compatible(a, b) for a, b in combinations(sorted(others), 2)):
            ambiguous.add(root)
    report['history_candidates_with_ambiguous_targets'] = len(ambiguous)
    report['history_supported_candidate_edges'] = len(profile_edges)
    for a, b, shared_dates, overlap, jaccard in sorted(profile_edges,
            key=lambda t: (-t[2], -t[3], -t[4], ids[t[0]], ids[t[1]])):
        if strong_only or find(a) in ambiguous or find(b) in ambiguous:
            review_pairs.append((ids[a], ids[b], 'strong_only_mode' if strong_only else 'ambiguous_identity_targets',
                                 shared_dates, round(overlap, 4), round(jaccard, 4)))
        else:
            ra, rb = find(a), find(b)
            if ra != rb and merge(a, b, 'name_and_nearly_identical_history',
                    f'shared_dates={shared_dates};overlap={overlap:.4f};jaccard={jaccard:.4f}'):
                root = find(a)
                other = rb if root == ra else ra
                history[root] |= history.pop(other)
    del candidate_pairs, blocks, name_groups, history
    gc.collect()

    print('3/5 · Единые профили и таблица соответствий ID', flush=True)
    def information(i):
        r = rows[i]
        return (10 * bool(passport[i]) + 5 * bool(birth[i]) +
                sum(bool(r.get(c, '')) for c in ['first_name', 'last_name', 'middle_name', 'full_name', 'sex']))
    user_mapping = {}
    merged_rows = []
    mapping_rows = []
    attribute_variants = []
    all_columns = list(u.columns)
    for root in sorted(members):
        group = members[root]
        order = sorted(group, key=lambda i: (-information(i), ids[i]))
        winner = order[0]
        canonical_id = ids[winner]
        merged = dict(rows[winner])
        for col in all_columns:
            values = list(dict.fromkeys(rows[i].get(col, '') for i in order if rows[i].get(col, '')))
            if not merged.get(col, '') and values:
                merged[col] = values[0]
            if col not in ['user_id', 'user_needs_review'] and len(values) > 1:
                attribute_variants.append((canonical_id, col, json.dumps(values, ensure_ascii=False)))
        if 'user_needs_review' in merged:
            merged['user_needs_review'] = 'true' if any(rows[i].get('user_needs_review', '').lower() == 'true' for i in group) else 'false'
        merged['user_id'] = canonical_id
        merged_rows.append(merged)
        for i in group:
            user_mapping[ids[i]] = canonical_id
            mapping_rows.append((ids[i], canonical_id, len(group), i == winner))
    result_users = pd.DataFrame(merged_rows, columns=all_columns).sort_values('user_id')
    save(result_users, out / 'users.csv')
    mapping_frame = pd.DataFrame(mapping_rows, columns=['old_user_id', 'canonical_user_id', 'group_size', 'is_representative'])
    if len(previous_id_map):
        historical = previous_id_map.copy()
        historical['canonical_user_id'] = historical.canonical_user_id.map(user_mapping)
        historical = historical[historical.canonical_user_id.notna()]
        mapping_frame = pd.concat([mapping_frame, historical], ignore_index=True).drop_duplicates('old_user_id')
        mapping_frame['group_size'] = mapping_frame.groupby('canonical_user_id').old_user_id.transform('size')
        mapping_frame['is_representative'] = mapping_frame.old_user_id.eq(mapping_frame.canonical_user_id)
    save(mapping_frame, audit / 'user_id_map.csv')
    save(pd.DataFrame(accepted, columns=['user_a', 'user_b', 'rule', 'evidence']), audit / 'accepted_merges.csv')
    save(pd.DataFrame(blocked, columns=['user_a', 'user_b', 'rule', 'conflict']), audit / 'blocked_merges.csv')
    save(pd.DataFrame(review_pairs, columns=['user_a', 'user_b', 'reason', 'shared_dates', 'overlap_smaller_history', 'jaccard']), audit / 'name_candidates_not_merged.csv')
    save(pd.DataFrame(attribute_variants, columns=['canonical_user_id', 'column', 'all_nonempty_values']), audit / 'user_attribute_variants.csv')
    # Все варианты исходных профилей в реально объединённых группах.
    merged_ids = {old for old, canonical, size, _ in mapping_rows if size > 1}
    new_archived = u[u.user_id.isin(merged_ids)]
    if len(archived_profiles) and len(new_archived):
        saved_profiles = pd.concat([archived_profiles, new_archived], ignore_index=True)
    else:
        saved_profiles = archived_profiles if len(archived_profiles) else new_archived
    save(saved_profiles.drop_duplicates(), audit / 'original_merged_user_profiles.csv')
    remaining_passports = result_users[result_users.passport_key.ne('')].copy()
    norm_p = remaining_passports.passport_key.map(normalize_document)
    save(remaining_passports[norm_p.duplicated(keep=False)], audit / 'remaining_shared_passports.csv')
    report['users_merged_away'] = n - len(result_users)
    report['merge_rules'] = dict(reasons)
    report['blocked_merge_attempts'] = len(blocked)
    report['name_candidates_not_merged'] = len(review_pairs)
    report['remaining_shared_passport_rows'] = int(norm_p.duplicated(keep=False).sum())

    print('4/5 · Перенос связей лояльности и перелётов', flush=True)
    loyalty['user_id'] = loyalty.user_id.map(user_mapping)
    loyalty = loyalty.drop_duplicates()
    save(loyalty, out / 'user_loyalty.csv')
    x['user_id'] = x.user_id.map(user_mapping)
    keys = ['user_id', 'flight_id']
    metadata = [c for c in ['observation_count', 'needs_review'] if c in x]
    business = [c for c in x if c not in keys + metadata]
    # Сначала сводим одинаковое содержимое. observation_count НЕ суммируется:
    # одни исходные наблюдения могли попасть в несколько профилей.
    original_repeated = x.duplicated(keys, keep=False)
    save(x.loc[original_repeated], audit / 'original_repeated_user_flights_after_remap.csv')
    agg = {}
    if 'observation_count' in x:
        x['observation_count'] = pd.to_numeric(x.observation_count, errors='raise').astype('int64')
        agg['observation_count'] = 'max'
    if 'needs_review' in x:
        x['needs_review'] = x.needs_review.str.lower().eq('true')
        agg['needs_review'] = 'max'
    if agg:
        events = x.groupby(keys + business, dropna=False, sort=False, as_index=False).agg(agg)
    else:
        events = x.drop_duplicates(keys + business)
    duplicates = events.duplicated(keys, keep=False)
    singles = events.loc[~duplicates].copy()
    repeated = events.loc[duplicates].copy()
    conflict_group_count = 0
    if len(repeated):
        nulls = repeated[keys + business].replace('', pd.NA)
        group = nulls.groupby(keys, sort=False, dropna=False)
        differing = group[business].nunique(dropna=True)
        conflict_index = differing.index[differing.gt(1).any(axis=1)]
        conflict_group_count = len(conflict_index)
        event_index = pd.MultiIndex.from_frame(repeated[keys])
        conflicting = repeated.loc[event_index.isin(conflict_index)].copy()
        # Разные билеты/классы/другие непустые сведения не выбрасываем наугад.
        if 'needs_review' in conflicting:
            conflicting['needs_review'] = True
        save(conflicting, audit / 'conflicting_user_flight_records.csv')
        complementary = repeated.loc[~event_index.isin(conflict_index)].copy()
        if len(complementary):
            tmp = complementary.replace('', pd.NA)
            rules = {c: 'first' for c in business}
            rules.update({c: 'max' for c in metadata})
            complementary = tmp.groupby(keys, sort=False, dropna=False, as_index=False).agg(rules).fillna('')
        events = pd.concat([singles, complementary, conflicting], ignore_index=True)
    if 'needs_review' in events:
        events['needs_review'] = events.needs_review.map({True: 'true', False: 'false'})
    if 'observation_count' in events:
        events['observation_count'] = pd.to_numeric(events.observation_count).astype('int64')
    events = events[list(x.columns)].sort_values(keys, kind='stable')
    save(events, out / 'user_flights.csv')
    save(f, out / 'flights.csv')
    report['conflicting_user_flight_groups_retained'] = conflict_group_count
    report['unique_user_flight_events'] = len(events[keys].drop_duplicates())
    report['output'] = {'users': len(result_users), 'flights': len(f),
                        'user_flights': len(events), 'user_loyalty': len(loyalty)}
    assert result_users.user_id.is_unique
    assert events.user_id.isin(result_users.user_id).all()
    assert events.flight_id.isin(f.flight_id).all()
    assert loyalty.user_id.isin(result_users.user_id).all()
    assert len(accepted) == n - len(result_users)
    # Проверяем именно сохранённые файлы, а не только таблицы в памяти.
    for table, expected_count in report['output'].items():
        actual_count = sum(len(chunk) for chunk in pd.read_csv(out / (table + '.csv'), sep=';',
            dtype='string', keep_default_na=False, usecols=[0], chunksize=100000))
        if actual_count != expected_count:
            raise ValueError(f'{table}: сохранено {actual_count} строк вместо {expected_count}')
    print('5/5 · Проверки и отчёт', flush=True)
    report['checks'] = {'unique_user_ids': True, 'unique_flight_ids': bool(f.flight_id.is_unique),
                         'valid_user_foreign_keys': True, 'valid_flight_foreign_keys': True}
    (out / 'deduplication_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f'Готово: {out}', flush=True)
    return report


def finalize_passes(passes, out):
    """Сводит журналы всех проходов и переносит стабильный результат."""
    out = Path(out).resolve()
    last = Path(passes[-1])
    reports = [json.loads((Path(p) / 'deduplication_report.json').read_text(encoding='utf-8')) for p in passes]
    final_map = read(last / 'audit', 'user_id_map').set_index('old_user_id').canonical_user_id.to_dict()
    accepted_parts = []
    variants = defaultdict(list)
    flight_maps = {}
    for step, folder in enumerate(passes, 1):
        folder = Path(folder)
        a = read(folder / 'audit', 'accepted_merges')
        if len(a):
            a['pass_number'] = step
            accepted_parts.append(a)
        for row in read(folder / 'audit', 'user_attribute_variants').itertuples(index=False):
            target = final_map.get(row.canonical_user_id, row.canonical_user_id)
            key = (target, row.column)
            for value in json.loads(row.all_nonempty_values):
                if value not in variants[key]:
                    variants[key].append(value)
        for row in read(folder / 'audit', 'flight_id_map').itertuples(index=False):
            flight_maps[row.old_flight_id] = row.canonical_flight_id
    accepted = pd.concat(accepted_parts, ignore_index=True) if accepted_parts else pd.DataFrame(
        columns=['user_a', 'user_b', 'rule', 'evidence', 'pass_number'])
    save(accepted, last / 'audit' / 'accepted_merges.csv')
    save(pd.DataFrame([(uid, col, json.dumps(values, ensure_ascii=False)) for (uid, col), values in variants.items()],
                     columns=['canonical_user_id', 'column', 'all_nonempty_values']), last / 'audit' / 'user_attribute_variants.csv')
    for old in list(flight_maps):
        target = flight_maps[old]
        seen = {old}
        while target in flight_maps and target not in seen:
            seen.add(target)
            target = flight_maps[target]
        flight_maps[old] = target
    save(pd.DataFrame(flight_maps.items(), columns=['old_flight_id', 'canonical_flight_id']), last / 'audit' / 'flight_id_map.csv')
    # Полные исходные сведения о повторных событиях первого прохода остаются в журнале.
    if len(passes) > 1:
        for filename in ['input_shared_passports.csv', 'original_repeated_user_flights_after_remap.csv']:
            shutil.copyfile(Path(passes[0]) / 'audit' / filename, last / 'audit' / filename)
    report = dict(reports[-1])
    report['input'] = reports[0]['input']
    report['users_merged_away'] = report['input']['users'] - report['output']['users']
    counts = Counter()
    for r in reports:
        counts.update(r['merge_rules'])
    report['merge_rules'] = dict(counts)
    report['exact_flight_ids_merged'] = len(flight_maps)
    report['passes'] = [{'pass': i, 'users_before': r['input']['users'], 'users_after': r['output']['users'],
                         'merged_ids': r['users_merged_away']} for i, r in enumerate(reports, 1)]
    report['stabilized_no_more_merges'] = reports[-1]['users_merged_away'] == 0
    assert len(accepted) == report['users_merged_away']
    (last / 'deduplication_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.move(str(last), str(out))
    print('ИТОГ:', json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def run(src, out, strong_only=False):
    src, out = Path(src).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Выходная папка уже существует. Укажите новую.')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Рабочие проходы находятся на том же диске; по завершении временные копии удаляются.
    with tempfile.TemporaryDirectory(prefix='.people_dedup_', dir=out.parent) as temp:
        passes = []
        current = src
        for number in range(1, 11):
            print(f'\nПРОХОД {number}: проверяю оставшиеся совпадения', flush=True)
            target = Path(temp) / f'pass_{number}'
            report = _run_pass(current, target, strong_only)
            passes.append(target)
            if report['users_merged_away'] == 0:
                return finalize_passes(passes, out)
            current = target
        raise ValueError('За 10 проходов объединение не стабилизировалось; исходники не изменены.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output')
    p.add_argument('--check-input', action='store_true', help='Проверить только заголовки и разделители четырёх CSV')
    p.add_argument('--strong-only', action='store_true', help='Отключить эвристику по историям поездок')
    a = p.parse_args()
    try:
        if a.check_input:
            for name in REQUIRED_COLUMNS:
                path, sep, encoding, skiprows, columns = csv_format(Path(a.input), name)
                print(f'{path.name}: разделитель={sep!r}, кодировка={encoding}, столбцы={columns!r}')
            print('Заголовки четырёх таблиц прошли проверку. Можно запускать очистку.')
            return
        if not a.output:
            p.error('Укажите --output, например --output cleaned_v2')
        run(a.input, a.output, a.strong_only)
    except (ValueError, KeyError, FileNotFoundError, UnicodeError, pd.errors.ParserError) as exc:
        p.exit(1, f'Ошибка: {exc}\nДля повторного запуска после ошибки укажите новую выходную папку.\n')


if __name__ == '__main__':
    main()
