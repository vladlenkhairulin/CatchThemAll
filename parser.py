"""
Читаем один источник целиком, собираем строки, сохраняем, переходим к другому.
Запуск: python parse_airline_sources_simple.py --json input.json --xml input.xml
Полные файлы должны быть корректно закрыты. Старый скрипт не используется.
"""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import tempfile
import uuid
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq


# Это названия колонок, а не сами данные. Ключи словаря — названия таблиц.
SCHEMAS = {
    "source_records": "source_record_id source_file source_type source_sheet record_type".split(),
    "passengers": "passenger_key passenger_name first_name middle_name last_name birth_date passenger_document".split(),
    "passenger_source_records": "passenger_key source_record_id passenger_id user_id".split(),
    "flights": "flight_key flight_date flight_number departure_time arrival_time departure_code arrival_code departure_city arrival_city airline flight_status".split(),
    "flight_source_records": "flight_key source_record_id flight_id flight_number_raw".split(),
    "passenger_flights": "source_record_id passenger_key flight_key flight_date flight_number booking_code ticket_number seat travel_class fare baggage agent_info gate sequence".split(),
    "loyalty_accounts": "ff_program ff_number".split(),
    "flight_loyalty": "source_record_id flight_key ff_program ff_number flight_date flight_number travel_class fare".split(),
}
REQUIRED = {
    "source_records": ["source_record_id", "source_file", "source_type", "record_type"],
    "passengers": ["passenger_key"],
    "passenger_source_records": ["passenger_key", "source_record_id"],
    "flights": ["flight_key", "flight_number"],
    "flight_source_records": ["flight_key", "source_record_id"],
    "passenger_flights": ["source_record_id"],
    "loyalty_accounts": ["ff_program", "ff_number"],
    "flight_loyalty": ["source_record_id", "ff_program", "ff_number"],
}


def empty_tables():
    """Для каждой таблицы создаём отдельный пустой список строк."""
    tables = {}
    for name in SCHEMAS:
        tables[name] = []
    return tables


def text(value):
    """Не обрезаем коды и не теряем нули. Пустое значение -> None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Ожидалась строка или NULL, получено: " + type(value).__name__)
    if value.strip() == "":
        return None
    return value


def parse_date(value, errors, location):
    """Проверяем ISO-дату; ошибочную дату отмечаем и заменяем на None."""
    value = text(value)
    if value is None:
        return None
    try:
        result = date.fromisoformat(value).isoformat()
        if result != value:
            raise ValueError("Нужен формат YYYY-MM-DD")
        return result
    except ValueError:
        errors.append({"kind": "invalid_date", "location": location})
        return None


def add_row(tables, table_name, **values):
    """Собираем строку строго по шаблону; отсутствующие поля -> None."""
    if set(values) - set(SCHEMAS[table_name]):
        raise ValueError("Лишние колонки для таблицы " + table_name)
    row = {}
    for field in SCHEMAS[table_name]:
        row[field] = values.get(field)
    row_index = len(tables[table_name])
    tables[table_name].append(row)
    return row_index


def add_source(tables, source, record_type):
    """Новый случайный ID исходной записи; общие ключи не создаём."""
    sid = str(uuid.uuid4())
    add_row(tables, "source_records", source_record_id=sid,
            source_file=source["file"], source_type=source["type"], record_type=record_type)
    return sid


def write_lineage(log, sid, location, **positions):
    """Сохраняем происхождение строк отдельно, не меняя колонки таблиц."""
    entry = {"source_record_id": sid, "location": location, **positions}
    log.write(json.dumps(entry, ensure_ascii=False) + "\n")


def add_passenger(tables, source, log, location, first, last, user_id=None, document=None):
    sid = add_source(tables, source, "passenger_profile")
    row = add_row(tables, "passengers", first_name=text(first),
                  last_name=text(last), passenger_document=text(document))
    add_row(tables, "passenger_source_records", source_record_id=sid, user_id=text(user_id))
    write_lineage(log, sid, location, passenger_row=row)
    return row


def add_account(tables, source, log, location, passenger_row, program, number, seen):
    """Объединяем только точные пары. seen — обычный словарь, не база."""
    sid = add_source(tables, source, "loyalty_account")
    program, number = text(program), text(number)
    key = (program, number)
    if program is not None and number is not None and key in seen:
        reference = seen[key]
    else:
        row = add_row(tables, "loyalty_accounts", ff_program=program, ff_number=number)
        reference = {"part": source["type"] + ".parquet", "row": row}
        if program is not None and number is not None:
            seen[key] = reference
    write_lineage(log, sid, location, passenger_row=passenger_row, loyalty_account=reference)
    return reference


def add_flight(tables, source, log, location, passenger_row, flight,
               user_id=None, account=None):
    """Один перелёт даёт строки в нескольких таблицах с одним source_record_id."""
    kind = "flight_loyalty" if account is not None else "passenger_flight"
    sid = add_source(tables, source, kind)
    row = add_row(tables, "flights", flight_date=flight["flight_date"],
                  flight_number=flight["flight_number"], departure_code=flight.get("departure_code"),
                  arrival_code=flight.get("arrival_code"), departure_city=flight.get("departure_city"),
                  arrival_city=flight.get("arrival_city"))
    add_row(tables, "flight_source_records", source_record_id=sid,
            flight_number_raw=flight["flight_number"])
    add_row(tables, "passenger_source_records", source_record_id=sid, user_id=text(user_id))
    add_row(tables, "passenger_flights", source_record_id=sid, flight_date=flight["flight_date"],
            flight_number=flight["flight_number"], fare=flight.get("fare"))
    if account is not None:
        add_row(tables, "flight_loyalty", source_record_id=sid,
                ff_program=account["program"], ff_number=account["number"],
                flight_date=flight["flight_date"], flight_number=flight["flight_number"], fare=flight.get("fare"))
    write_lineage(log, sid, location, passenger_row=passenger_row, flight_row=row,
                  loyalty_account=account["reference"] if account is not None else None)


def parse_json(path, log, errors, seen):
    """Открываем JSON целиком и обычными циклами разбираем профили."""
    with path.open(encoding="utf-8-sig") as file:
        data = json.load(file, parse_int=str, parse_float=str)
    profiles = data["Forum Profiles"]
    if not isinstance(profiles, list):
        raise ValueError("Forum Profiles должен быть массивом")
    tables = empty_tables()
    source = {"file": path.name, "type": "json"}
    for i, profile in enumerate(profiles):
        location = f"$.Forum Profiles[{i}]"
        name = profile.get("Real Name") or {}
        passports = []
        for document in profile.get("Travel Documents") or []:
            passport = text(document.get("Passports"))
            if passport is not None:
                passports.append(passport)
        if len(passports) > 1:
            raise ValueError(location + ": несколько паспортов не помещаются в одно поле")
        document = passports[0] if passports else None
        passenger_row = add_passenger(tables, source, log, location,
                                      name.get("First Name"), name.get("Last Name"), document=document)
        for c, card in enumerate(profile.get("Loyality Programm") or []):
            add_account(tables, source, log, f"{location}.Loyality Programm[{c}]",
                        passenger_row, card.get("programm"), card.get("Number"), seen)
        for f, item in enumerate(profile.get("Registered Flights") or []):
            flight_location = f"{location}.Registered Flights[{f}]"
            departure = item.get("Departure") or {}
            arrival = item.get("Arrival") or {}
            flight = {
                "flight_number": text(item.get("Flight")),
                "flight_date": parse_date(item.get("Date"), errors, flight_location + ".Date"),
                "departure_code": text(departure.get("Airport")),
                "arrival_code": text(arrival.get("Airport")),
                "departure_city": text(departure.get("City")),
                "arrival_city": text(arrival.get("City")),
            }
            add_flight(tables, source, log, flight_location, passenger_row, flight)
    # data/profiles — локальные переменные: после return дерево JSON больше не нужно.
    return tables


def parse_xml(path, log, errors, seen):
    """Строим XML-дерево целиком; перебираем пользователей, карты, перелёты."""
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "PointzAggregatorUsers":
        raise ValueError("Неверный корневой элемент XML")
    tables = empty_tables()
    source = {"file": path.name, "type": "xml"}
    for i, user in enumerate(root):
        if user.tag != "user":
            raise ValueError("Ожидался элемент user")
        location = f"/PointzAggregatorUsers/user[{i + 1}]"
        name = user.find("name")
        first = name.get("first") if name is not None else None
        last = name.get("last") if name is not None else None
        uid = user.get("uid")
        passenger_row = add_passenger(tables, source, log, location, first, last, user_id=uid)
        for g, cards in enumerate(user.findall("cards")):
            if cards.get("type") != "Airlines":
                raise ValueError(location + ": ожидался cards type=Airlines")
            for c, card in enumerate(cards.findall("card")):
                card_location = f"{location}/cards[{g + 1}]/card[{c + 1}]"
                program = text(card.findtext("bonusprogramm"))
                number = text(card.get("number"))
                reference = add_account(tables, source, log, card_location,
                                        passenger_row, program, number, seen)
                account = {"program": program, "number": number, "reference": reference}
                for ag, activities in enumerate(card.findall("activities")):
                    for a, activity in enumerate(activities.findall("activity")):
                        flight_location = f"{card_location}/activities[{ag + 1}]/activity[{a + 1}]"
                        if activity.get("type") != "Flight":
                            errors.append({"kind": "unsupported_activity", "location": flight_location})
                            continue
                        flight = {
                            "flight_number": text(activity.findtext("Code")),
                            "flight_date": parse_date(activity.findtext("Date"), errors, flight_location + "/Date"),
                            "departure_code": text(activity.findtext("Departure")),
                            "arrival_code": text(activity.findtext("Arrival")),
                            "fare": text(activity.findtext("Fare")),
                        }
                        add_flight(tables, source, log, flight_location, passenger_row,
                                   flight, user_id=uid, account=account)
    return tables


def save_tables(tables, part, staging, errors):
    """Записываем один файл на таблицу для текущего источника и считаем NULL."""
    report = {"row_counts": {}, "null_counts": {}, "deferred_required_keys": {}}
    for name, rows in tables.items():
        fields = SCHEMAS[name]
        nulls = {field: 0 for field in fields}
        for i, row in enumerate(rows):
            for field in fields:
                if row[field] is None:
                    nulls[field] += 1
                    if field in REQUIRED[name] and field not in ("passenger_key", "flight_key"):
                        errors.append({"kind": "missing_required", "table": name,
                                       "part": part, "row": i, "field": field})
        schema = pa.schema([pa.field(field, pa.string(), nullable=True) for field in fields])
        table = pa.Table.from_pylist(rows, schema=schema)
        folder = staging / name
        folder.mkdir(exist_ok=True)
        pq.write_table(table, folder / part, compression="snappy")
        # Не читаем каждый файл обратно: эту проверку оставили отдельным тестам.
        report["row_counts"][name] = len(rows)
        report["null_counts"][name] = nulls
        report["deferred_required_keys"][name] = {
            field: nulls[field] for field in REQUIRED[name]
            if field in ("passenger_key", "flight_key")
        }
        # Эта таблица сохранена: её список можно очистить до сохранения следующей.
        rows.clear()
        del table
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path("FrequentFlyerForum-Profiles.json"))
    parser.add_argument("--xml", type=Path, default=Path("PointzAggregator-AirlinesData.xml"))
    parser.add_argument("--output", type=Path, default=Path("staging_simple"))
    args = parser.parse_args()
    output = args.output.resolve()
    audit = output.with_name(output.name + "_audit")
    if output.exists() or audit.exists():
        parser.error("Папка результата/аудита уже есть. Укажи новый --output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_accounts = {}  # Только точные пары (программа, номер); без нормализации.
    summary = {"status": "ok", "sources": {}}
    # Временная папка не даёт принять незавершённую выгрузку за готовую.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".simple-parser-") as temp:
        staging, logs = Path(temp) / "staging", Path(temp) / "audit"
        staging.mkdir()
        logs.mkdir()
        for kind, path, read in [("json", args.json, parse_json), ("xml", args.xml, parse_xml)]:
            errors = []
            report = {"source_file": path.name, "part": kind + ".parquet"}
            try:
                with (logs / (kind + "_lineage.jsonl")).open("w", encoding="utf-8") as log:
                    tables = read(path, log, errors, seen_accounts)
                report.update(save_tables(tables, kind + ".parquet", staging, errors))
                del tables  # JSON-таблицы не храним одновременно с XML-деревом.
            except Exception as exc:
                errors.append({"kind": "fatal_error", "detail": str(exc)})
                summary["status"] = "failed"
            with (logs / (kind + "_issues.jsonl")).open("w", encoding="utf-8") as log:
                for error in errors:
                    log.write(json.dumps(error, ensure_ascii=False) + "\n")
            report["issue_counts"] = dict(Counter(error["kind"] for error in errors))
            summary["sources"][kind] = report
            print(kind.upper(), "строки:", report.get("row_counts", {}))
            print("Ошибки:", report["issue_counts"])
            if summary["status"] == "failed":
                break
            if errors:
                summary["status"] = "ok_with_issues"
        (logs / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logs.rename(audit)
        if summary["status"] != "failed":
            staging.rename(output)
    print("Аудит:", audit)
    if summary["status"] == "failed":
        print("Ошибка: итоговая папка таблиц не создана. Смотри аудит.")
        return 1
    print("Таблицы:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
