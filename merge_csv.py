"""Merge prepared CSVs. Python 3.10+. No handwritten SQL. Run in PyCharm.

Input: a completed conversion folder. Output: users, flights, user_flights and user_loyalty CSV tables.
Identity: passport first, loyalty program+number only when passport is absent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from functools import lru_cache
from contextlib import ExitStack
from itertools import zip_longest
from disk_store import DiskStore, ordered_records
from pathlib import Path

# Leave None to select the latest FULL, completed run inside converted/.
# Or specify: INPUT_DIR = Path(r"C:\...\converted\run_...")
INPUT_DIR = None
CONVERTED_DIR = Path(__file__).resolve().parent / "converted"
OUTPUT_DIR = Path(__file__).resolve().parent / "merged"
# Rich passport sources first; anonymous loyalty-only observations last.
SOURCES = ("sirena", "boarding", "pointz", "forum", "boardingpass", "skyteam")
PERSON_FIELDS = ("first_name", "middle_name", "last_name", "full_name", "birth_date_raw", "birth_date_iso", "sex", "passport_raw", "passport_key")
DATA_FIELDS = [
    "record_kind", *PERSON_FIELDS, "loyalty_cards_json", "flight_date", "flight_time", "flight_number",
    "departure_airport", "departure_city", "departure_country", "arrival_airport", "arrival_city", "arrival_country",
    "arrival_date", "arrival_time", "booking_code", "ticket_number", "codeshare_raw", "travel_class", "fare",
    "baggage_status", "baggage_allowance", "seat", "meal", "agent", "flight_status", "gate", "boarding_sequence", "honorific", "issues_json",
]
MISSING = {"", "n/a", "not presented", "null"}
TRANSLIT = dict(zip("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
    ["A","B","V","G","D","E","E","ZH","Z","I","I","K","L","M","N","O","P","R","S","T","U","F","KH","TS","CH","SH","SHCH","","Y","","E","IU","IA"]))
csv.field_size_limit(64 * 1024 * 1024)


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter=";")


def writer(stream, columns):
    result = csv.DictWriter(stream, fieldnames=columns, delimiter=";", extrasaction="ignore")
    result.writeheader()
    return result


def present(value):
    return value is not None and str(value).strip().casefold() not in MISSING


def document(value):
    if not present(value) or str(value).strip().casefold() == "none":
        return ""
    result = re.sub(r"[\s-]", "", unicodedata.normalize("NFKC", value)).upper()
    return result if re.fullmatch(r"[A-ZА-ЯЁ0-9]+", result) else ""


def cards(row):
    values = json.loads(row.get("loyalty_cards_json") or "[]")
    if not isinstance(values, list):
        raise ValueError("loyalty_cards_json must contain an array")
    result = {}
    for value in values:
        program = str(value.get("program_key") or value.get("program_raw") or "").strip().upper()
        number = re.sub(r"\s", "", str(value.get("number_key") or value.get("number_raw") or ""))
        if re.fullmatch(r"[A-Z0-9]{2,8}", program) and re.fullmatch(r"\d+", number):
            result[f"{program}:{number}"] = value
    return result


@lru_cache(maxsize=100000)
def name_token(value):
    value = unicodedata.normalize("NFKC", value).upper()
    return "".join(TRANSLIT.get(c, c) for c in value if c.isalpha())


def token_equal(a, b):
    a, b = name_token(a), name_token(b)
    if not a or not b:
        return True
    if min(len(a), len(b)) == 1:
        return a[0] == b[0]
    if a == b:
        return True
    # Common spelling variants; used ONLY after an identifier has matched.
    return spelling(a) == spelling(b)


def spelling(s):
    s = s.replace("SHCH", "SCH").replace("KH", "H").replace("X", "KS")
    s = s.replace("YU", "IU").replace("YA", "IA").replace("YO", "E").replace("Y", "I")
    return re.sub("I+", "I", s)


def name_options(p):
    if p.get("first_name") or p.get("last_name"):
        return [(p.get("first_name", ""), p.get("last_name", ""), p.get("middle_name", ""))]
    parts = re.findall(r"[^\W\d_]+", p.get("full_name", ""), re.UNICODE)
    if len(parts) == 2:
        return [(parts[0], parts[1], ""), (parts[1], parts[0], "")]
    if len(parts) == 3:
        return [(parts[1], parts[0], parts[2]), (parts[0], parts[2], parts[1])]
    return []


def compatible(a, b):
    pa, pb = a.get("passport_key", ""), b.get("passport_key", "")
    if pa and pb and pa != pb:
        return False
    ba, bb = a.get("birth_date_iso", ""), b.get("birth_date_iso", "")
    if ba and bb and ba != bb:
        return False
    sa, sb = a.get("sex", ""), b.get("sex", "")
    if present(sa) and present(sb) and sa.casefold() != sb.casefold():
        return False
    na, nb = name_options(a), name_options(b)
    if not na or not nb:
        return True
    return any(all(token_equal(x, y) for x, y in zip(va, vb)) for va in na for vb in nb)


def has_name(p):
    return bool(name_options(p))


def strong_name(p):
    options = name_options(p)
    if not options:
        return ""
    first,last,_ = options[0]
    first,last = spelling(name_token(first)),spelling(name_token(last))
    return dump(sorted((first,last))) if len(first)>1 and len(last)>1 else ""


def flight_signature(row):
    date = row.get("flight_date", "").strip()
    code = re.sub(r"\s", "", row.get("flight_number", "")).upper()
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return None
    if not re.fullmatch(r"[A-Z0-9]{2}\d{1,5}[A-Z]?", code):
        return None
    clock = row.get("flight_time", "").strip()
    if not re.fullmatch(r"\d{2}:\d{2}", clock):
        clock = ""
    airports = [row.get(k, "").strip() for k in ("departure_airport", "arrival_airport")]
    airports = [a if re.fullmatch("[A-Z]{3}", a) else "" for a in airports]
    return date, code, clock, *airports


def stable_flight_id(parts):
    return "F" + hashlib.sha256(dump(parts).encode()).hexdigest()[:24]


def choose_input(path):
    path = Path(path)
    def completed(folder):
        if (folder / "INCOMPLETE.txt").exists() or not (folder / "SUCCESS.csv").is_file():
            return False
        status = list(read_csv(folder / "SUCCESS.csv"))
        return bool(status and status[0].get("mode") == "full")
    if completed(path):
        return path
    if path.is_dir():
        candidates = sorted((p for p in path.iterdir() if p.is_dir() and completed(p)), reverse=True)
        if candidates:
            return candidates[0]
    raise ValueError(f"No completed FULL conversion found in {path}. SAMPLE/INCOMPLETE runs are excluded.")


def observations(folder, sources, limit=0):
    for source in sources:
        main = folder / f"{source}.csv"
        links = folder / f"{source}.links.csv"
        if not main.is_file() or not links.is_file():
            raise FileNotFoundError(f"Missing {main.name} or {links.name} in {folder}")
        print(f"Reading {source}...", flush=True)
        a, b = read_csv(main), read_csv(links)
        try:
            for index, pair in enumerate(zip_longest(a, b), 1):
                row, link = pair
                if row is None or link is None or not row.get("observation_id") or row["observation_id"] != link.get("observation_id"):
                    raise ValueError(f"Main/links mismatch in {source}, row {index}")
                if not link.get("source_subject_id") or not link.get("source_record_id"):
                    raise ValueError(f"Missing provenance in {source}, row {index}")
                yield source, row, link
                if limit and index >= limit:
                    break
        finally:
            a.close()
            b.close()


# Four output entities. No arrays/dictionaries are written into these tables.
USER_COLUMNS = ["user_id", "first_name", "middle_name", "last_name", "full_name", "birth_date_iso", "sex", "passport_key", "identity_status", "user_needs_review"]
FLIGHT_FIELDS = ["flight_date", "flight_time", "flight_number", "departure_airport", "departure_city", "departure_country", "arrival_airport", "arrival_city", "arrival_country", "arrival_date", "arrival_time", "flight_status", "gate", "codeshare_raw"]
FLIGHT_COLUMNS = ["flight_id", *[f for f in FLIGHT_FIELDS if f != "codeshare_raw"], "codeshare", "flight_needs_review"]
PARTICIPANT_FIELDS = ["record_kind", "booking_code", "ticket_number", "seat", "travel_class", "fare", "baggage_status", "baggage_allowance", "meal", "agent", "boarding_sequence", "honorific"]
PARTICIPANT_COLUMNS = ["user_id", "flight_id", *PARTICIPANT_FIELDS, "observation_count", "needs_review"]
LOYALTY_COLUMNS = ["user_id", "program_code", "card_number"]
CODESHARE = {"own":"false", "no":"false", "false":"false", "operated":"true", "yes":"true", "true":"true", "operated by some other airline":"true"}


def index_line(stream, value):
    stream.write(dump(value) + "\n")


def key_name(kind, value):
    return kind + ":" + value


def personal_profile(row):
    p = {field: row.get(field, "") for field in PERSON_FIELDS}
    p["passport_key"] = document(p["passport_key"] or p["passport_raw"])
    return {k: v if present(v) else "" for k,v in p.items()}


def name_middle(p):
    value = spelling(name_token(p.get("middle_name", "")))
    return value if len(value)>1 else ""


def collect(state, folder, sources, limit, service):
    counts = Counter()
    with (service/"subjects.order.jsonl").open("w",encoding="utf-8") as people, (service/"signatures.order.jsonl").open("w",encoding="utf-8") as signatures, (service/"keys.order.jsonl").open("w",encoding="utf-8") as keys:
        for source,row,link in observations(folder,sources,limit):
            sid = link["source_subject_id"]
            if state.get("observation",row["observation_id"]) is not None:
                raise ValueError(f"Duplicate observation ID: {row['observation_id']}")
            state.put("observation",row["observation_id"],sid)
            p = personal_profile(row)
            subject = state.get("subject",sid)
            if subject is None:
                subject = {"profile":p,"keys":set(),"uid":None,"status":""}
                rich = bool(p["passport_key"])*10000 + has_name(p)*100 + bool(strong_name(p))*50 + bool(p["birth_date_iso"])*10 + min(len(name_middle(p)),4) + bool(any(p.values()))
                index_line(people,[-rich,sid])
            elif subject["profile"] != p:
                raise ValueError(f"Contradicting personal fields within source profile {sid}")
            claims = [("passport",p["passport_key"])] if p["passport_key"] else []
            claims += [("loyalty",card_key) for card_key in cards(row)]
            for kind,value in claims:
                address = key_name(kind,value)
                evidence = state.get("key",address)
                if evidence is None:
                    evidence = {"kind":kind,"value":value,"names":set(),"births":set(),"middles":set(),"sexes":set(),"owners":set()}
                    index_line(keys,address)
                for field,v in (("names",strong_name(p)),("births",p["birth_date_iso"]),("middles",name_middle(p)),("sexes",p["sex"].casefold())):
                    if v:
                        evidence[field].add(v)
                state.put("key",address,evidence)
                subject["keys"].add(address)
            state.put("subject",sid,subject)
            sig = flight_signature(row) if row.get("record_kind", "user_flight") == "user_flight" else None
            if sig and state.get("signature",dump(sig)) is None:
                state.put("signature",dump(sig),{"fid":None,"ambiguous":False})
                index_line(signatures,[-sum(bool(x) for x in sig[2:]),dump(sig)])
            counts[source] += 1
            if sum(counts.values())%20000 == 0:
                state.commit()
                print(f"  Collected {sum(counts.values()):,} observations",flush=True)
    state.commit()
    return counts


def assign_users(state, service):
    print("Matching users: passports, then loyalty cards...",flush=True)
    decisions = Counter()
    user_count = 0
    for index,(_,sid) in enumerate(ordered_records(service/"subjects.order.jsonl",service/"sort_subjects"),1):
        subject = state.get("subject",sid)
        p = subject["profile"]
        lookup = [key_name("passport",p["passport_key"])] if p["passport_key"] else sorted(k for k in subject["keys"] if k.startswith("loyalty:"))
        if not lookup:
            subject["status"] = "excluded_no_identifier"
            state.put("subject",sid,subject)
            decisions["excluded_no_identifier"] += 1
            continue
        evidence = [state.get("key",k) for k in lookup]
        owners = set().union(*(e["owners"] for e in evidence))
        anonymous = not any(p.values())
        risks = {field for field in ("names","births","middles","sexes") if any(len(e[field])>1 for e in evidence)}
        candidates = []
        for uid in sorted(owners):
            user = state.get("user",uid)
            # A card-only bucket is not evidence of an identified owner.
            # In particular, it must never bridge two conflicting people.
            if user.get("unresolved_loyalty"):
                continue
            profiles = user["profiles"]
            if "births" in risks and (not p["birth_date_iso"] or not any(v["birth_date_iso"]==p["birth_date_iso"] for v in profiles)):
                continue
            if "names" in risks and (not strong_name(p) or not any(strong_name(v) for v in profiles)):
                continue
            if "middles" in risks and (not name_middle(p) or not any(name_middle(v) for v in profiles)):
                continue
            if "sexes" in risks and (not p["sex"] or not any(v["sex"] for v in profiles)):
                continue
            if all(compatible(p,v) for v in profiles):
                candidates.append((uid,user))
        if len(candidates)==1:
            uid,user = candidates[0]
            status = "passport_match" if p["passport_key"] else "loyalty_match"
            if p not in user["profiles"]:
                user["profiles"].append(p)
        elif anonymous:
            # Repeated anonymous claims with the EXACT same cards share one
            # unresolved bucket. A flight observation is not a new person.
            bucket_key = dump(sorted(subject["keys"]))
            uid = state.get("anonymous_bucket",bucket_key)
            if uid is None:
                user_count += 1
                uid = user_id(user_count)
                user = {"profiles":[p],"keys":set(),"statuses":set(),
                        "unresolved_loyalty":True,"reasons":{"card_only_identity"}}
                state.put("anonymous_bucket",bucket_key,uid)
                status = "anonymous_bucket_created"
            else:
                user = state.get("user",uid)
                status = "anonymous_bucket_reused"
            if len(candidates)>1:
                user["reasons"].add("ambiguous_separate")
            elif owners - {uid}:
                user["reasons"].add("conflict_separate")
        else:
            user_count += 1
            uid = user_id(user_count)
            user = {"profiles":[p],"keys":set(),"statuses":set()}
            status = "ambiguous_separate" if len(candidates)>1 else "conflict_separate" if owners else "new_identifier"
        user["keys"].update(subject["keys"])
        user["statuses"].add(status)
        state.put("user",uid,user)
        for address in subject["keys"]:
            key = state.get("key",address)
            key["owners"].add(uid)
            state.put("key",address,key)
        subject.update(uid=uid,status=status)
        state.put("subject",sid,subject)
        decisions[status] += 1
        if index%20000 == 0:
            state.commit()
            print(f"  Resolved {index:,} source profiles",flush=True)
    state.commit()
    return user_count,decisions


def assign_flights(state, service):
    print("Matching flight dates, numbers, routes and times...",flush=True)
    for index,(_,encoded) in enumerate(ordered_records(service/"signatures.order.jsonl",service/"sort_signatures"),1):
        date,code,clock,origin,destination = json.loads(encoded)
        address = dump([date,code])
        anchors = state.get("flight_anchors",address,[])
        options = [a["fid"] for a in anchors if all(not x or not y or x==y for x,y in zip((clock,origin,destination),a["details"]))]
        if len(options)>1:
            result = {"fid":None,"ambiguous":True}
        else:
            fid = options[0] if options else stable_flight_id((date,code,clock,origin,destination))
            if not options:
                anchors.append({"fid":fid,"details":(clock,origin,destination)})
                state.put("flight_anchors",address,anchors)
            result = {"fid":fid,"ambiguous":False}
        state.put("signature",encoded,result)
        if index%20000 == 0:
            state.commit()
            print(f"  Resolved {index:,} flight signatures",flush=True)
    state.commit()


def add_values(values, row, fields):
    for field in fields:
        value = row.get(field, "")
        if present(value) and value not in values.setdefault(field,[]):
            values[field].append(value)


def combine_observations(state, folder, sources, limit, service):
    print("Collecting flights and passenger participation...",flush=True)
    counts = Counter()
    with ExitStack() as stack:
        excluded = writer(stack.enter_context((service/"excluded.csv").open("w",encoding="utf-8-sig",newline="")),["source","observation_id","source_subject_id","reason"])
        no_flight = writer(stack.enter_context((service/"profile_only.csv").open("w",encoding="utf-8-sig",newline="")),["source","observation_id","user_id","record_kind"])
        provenance = writer(stack.enter_context((service/"provenance.csv").open("w",encoding="utf-8-sig",newline="")),["observation_id","source","source_subject_id","source_record_id","user_id","flight_id","status","identity_decision"])
        flights_index = stack.enter_context((service/"flights.order.jsonl").open("w",encoding="utf-8"))
        pairs_index = stack.enter_context((service/"participation.order.jsonl").open("w",encoding="utf-8"))
        for index,(source,row,link) in enumerate(observations(folder,sources,limit),1):
            sid = link["source_subject_id"]
            subject = state.get("subject",sid)
            uid = subject["uid"] or ""
            record = dict(observation_id=row["observation_id"],source=source,source_subject_id=sid,source_record_id=link["source_record_id"],user_id=uid,flight_id="",identity_decision=subject["status"])
            if not uid:
                excluded.writerow(dict(source=source,observation_id=row["observation_id"],source_subject_id=sid,reason=subject["status"]))
                record["status"] = "excluded"
                counts["excluded"] += 1
            elif row.get("record_kind", "user_flight") != "user_flight":
                no_flight.writerow(dict(source=source,observation_id=row["observation_id"],user_id=uid,record_kind=row.get("record_kind","")))
                record["status"] = "profile_only"
                counts["profile_only"] += 1
            else:
                sig = flight_signature(row)
                match = state.get("signature",dump(sig)) if sig else None
                fid = match["fid"] if match else None
                reason = "ambiguous_flight" if match and match["ambiguous"] else "incomplete_flight" if not fid else ""
                if not fid:
                    fid = stable_flight_id(("observation",row["observation_id"]))
                flight = state.get("flight",fid)
                if flight is None:
                    flight = {"values":{},"reasons":set()}
                    index_line(flights_index,fid)
                add_values(flight["values"],row,FLIGHT_FIELDS)
                if reason:
                    flight["reasons"].add(reason)
                state.put("flight",fid,flight)
                pair_key = dump([uid,fid])
                pair = state.get("participation",pair_key)
                if pair is None:
                    pair = {"values":{},"count":0,"source_issues":set()}
                    index_line(pairs_index,[uid,fid])
                add_values(pair["values"],row,PARTICIPANT_FIELDS)
                pair["count"] += 1
                pair["source_issues"].update(json.loads(row.get("issues_json") or "[]"))
                state.put("participation",pair_key,pair)
                record.update(flight_id=fid,status="linked")
                counts["linked"] += 1
            provenance.writerow(record)
            if index%20000 == 0:
                state.commit()
                print(f"  Collected participation from {index:,} observations",flush=True)
    state.commit()
    return counts


def choose_values(values, entity, entity_id, alternatives_writer):
    result,conflicting = {},False
    for field,variants in values.items():
        if not variants:
            continue
        selected = variants[0]
        if field.endswith("_airport"):
            selected = next((v for v in variants if re.fullmatch("[A-Z]{3}",v)),selected)
        result[field] = selected
        if len(variants)>1:
            for value in variants:
                alternatives_writer.writerow(dict(entity_type=entity,entity_id=entity_id,field=field,value=value,selected=str(value==selected).lower()))
            canonical = {v.strip().casefold() for v in variants}
            if field=="codeshare_raw":
                canonical = {CODESHARE.get(v.strip().casefold(),v.strip().casefold()) for v in variants}
            if entity != "user" and len(canonical)>1:
                conflicting = True
    return result,conflicting


def export(state, output, user_count, user_index=None):
    print("Writing users.csv, flights.csv, user_flights.csv, user_loyalty.csv...",flush=True)
    service = output/"service"
    counts = Counter()
    with ExitStack() as stack:
        tables = {}
        for name,columns in (("users",USER_COLUMNS),("flights",FLIGHT_COLUMNS),("user_flights",PARTICIPANT_COLUMNS),("user_loyalty",LOYALTY_COLUMNS)):
            tables[name] = writer(stack.enter_context((output/f"{name}.csv").open("w",encoding="utf-8-sig",newline="")),columns)
        alternatives = writer(stack.enter_context((service/"field_values.csv").open("w",encoding="utf-8-sig",newline="")),["entity_type","entity_id","field","value","selected"])
        reviews = writer(stack.enter_context((service/"review_reasons.csv").open("w",encoding="utf-8-sig",newline="")),["entity_type","entity_id","reason"])
        conflicts = writer(stack.enter_context((service/"identifier_conflicts.csv").open("w",encoding="utf-8-sig",newline="")),["identifier_type","identifier_key","user_id"])
        ids = ordered_records(user_index,service/"sort_users") if user_index else (user_id(i) for i in range(1,user_count+1))
        for index,uid in enumerate(ids,1):
            user = state.get("user",uid)
            values = {}
            for p in user["profiles"]:
                add_values(values,p,PERSON_FIELDS)
            for field,variants in user.get("field_variants",{}).items():
                for value in variants:
                    if present(value) and value not in values.setdefault(field,[]):
                        values[field].append(value)
            person,unused = choose_values(values,"user",uid,alternatives)
            reasons = (set(user["statuses"]) & {"ambiguous_separate","conflict_separate"}) | set(user.get("reasons",()))
            for address in sorted(user["keys"]):
                evidence = state.get("key",address)
                if len(evidence["owners"])>1:
                    reasons.add("shared_"+evidence["kind"])
                    conflicts.writerow(dict(identifier_type=evidence["kind"],identifier_key=evidence["value"],user_id=uid))
                if evidence["kind"]=="loyalty":
                    program,number = evidence["value"].split(":",1)
                    tables["user_loyalty"].writerow(dict(user_id=uid,program_code=program,card_number=number))
                    counts["user_loyalty"] += 1
            user["needs_review"] = bool(reasons)
            state.put("user",uid,user)
            identity_status = "unresolved_loyalty" if user.get("unresolved_loyalty") else "needs_review" if reasons else "identifier_based"
            if user.get("unresolved_loyalty"):
                counts["unresolved_loyalty_profiles"] += 1
            tables["users"].writerow({**person,"user_id":uid,"identity_status":identity_status,"user_needs_review":str(bool(reasons)).lower()})
            counts["users"] += 1
            for reason in sorted(reasons):
                reviews.writerow(dict(entity_type="user",entity_id=uid,reason=reason))
            if index%20000==0:
                state.commit()
                print(f"  Exported {index:,} users",flush=True)
        for fid in ordered_records(service/"flights.order.jsonl",service/"sort_flights"):
            flight = state.get("flight",fid)
            values,conflicting = choose_values(flight["values"],"flight",fid,alternatives)
            reasons = set(flight["reasons"])
            if conflicting:
                reasons.add("conflicting_flight_values")
            raw = values.pop("codeshare_raw", "")
            codeshare = CODESHARE.get(raw.casefold(), "")
            if raw and not codeshare:
                reasons.add("unrecognized_codeshare")
            flight["needs_review"] = bool(reasons)
            state.put("flight",fid,flight)
            tables["flights"].writerow({**values,"flight_id":fid,"codeshare":codeshare,"flight_needs_review":str(bool(reasons)).lower()})
            counts["flights"] += 1
            for reason in sorted(reasons):
                reviews.writerow(dict(entity_type="flight",entity_id=fid,reason=reason))
            if counts["flights"]%20000==0:
                state.commit()
                print(f"  Exported {counts['flights']:,} flights",flush=True)
        for uid,fid in ordered_records(service/"participation.order.jsonl",service/"sort_participation"):
            pair = state.get("participation",dump([uid,fid]))
            user,flight = state.get("user",uid),state.get("flight",fid)
            if user is None or flight is None:
                raise RuntimeError("Broken user/flight foreign key")
            values,conflicting = choose_values(pair["values"],"participation",uid+"/"+fid,alternatives)
            tables["user_flights"].writerow({**values,"user_id":uid,"flight_id":fid,"observation_count":pair["count"],"needs_review":str(bool(user["needs_review"] or flight["needs_review"] or conflicting or pair.get("needs_review"))).lower()})
            counts["user_flights"] += 1
            counts["linked_observations"] += pair["count"]
            for issue in sorted(pair["source_issues"]):
                reviews.writerow(dict(entity_type="participation",entity_id=uid+"/"+fid,reason="source:"+issue))
            if conflicting:
                reviews.writerow(dict(entity_type="participation",entity_id=uid+"/"+fid,reason="conflicting_participation_values"))
            if pair.get("needs_review"):
                reviews.writerow(dict(entity_type="participation",entity_id=uid+"/"+fid,reason="legacy_needs_review"))
            if counts["user_flights"]%20000==0:
                state.commit()
                print(f"  Exported {counts['user_flights']:,} user-flight rows",flush=True)
    state.commit()
    return counts


def user_id(number):
    return f"U{number:09d}"


def run(input_dir, output_root, sources=SOURCES, limit_rows=0):
    if limit_rows<0:
        raise ValueError("limit_rows must be nonnegative")
    folder = choose_input(input_dir)
    files = [folder/f"{source}{suffix}.csv" for source in sources for suffix in ("", ".links")]
    snapshots = {p:(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    output = Path(output_root)/(datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")+("_SAMPLE" if limit_rows else ""))
    service = output/"service"
    service.mkdir(parents=True)
    marker = output/"INCOMPLETE.txt"
    marker.write_text("Incomplete merge; do not use these tables yet.\n",encoding="utf-8")
    state = DiskStore(service/"state.sqlite")
    print(f"Input: {folder}\nOutput: {output}",flush=True)
    try:
        input_counts = collect(state,folder,sources,limit_rows,service)
        users,decisions = assign_users(state,service)
        assign_flights(state,service)
        observation_counts = combine_observations(state,folder,sources,limit_rows,service)
        if any((p.stat().st_size,p.stat().st_mtime_ns)!=before for p,before in snapshots.items()):
            raise RuntimeError("Input files changed during merge")
        tables = export(state,output,users)
        total = sum(input_counts.values())
        if sum(observation_counts.values())!=total or tables["linked_observations"]!=observation_counts["linked"]:
            raise RuntimeError("Observation reconciliation failed")
        report = {"schema_version":"2.1","input_folder":str(folder.resolve()),"mode":"sample" if limit_rows else "full", "observations":total,"excluded_observations":observation_counts["excluded"],"profile_only_observations":observation_counts["profile_only"],"per_source":dict(input_counts),"subject_decisions":dict(decisions),**dict(tables)}
        state.put("metadata","report",report)
        state.commit()
        with (service/"summary.csv").open("w",encoding="utf-8-sig",newline="") as stream:
            w = writer(stream,["metric","value"])
            w.writerows({"metric":k,"value":dump(v)} for k,v in report.items())
        (output/"SUCCESS.txt").write_text(f"Schema: four tables\nMode: {report['mode']}\nObservations: {total}\n",encoding="utf-8")
        marker.unlink()
        print(f"SUCCESS: {output}",flush=True)
        return output
    finally:
        state.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=INPUT_DIR or CONVERTED_DIR)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT_DIR)
    parser.add_argument("--limit-rows",type=int,default=0,help="Per-source observation limit; 0 = full merge")
    args = parser.parse_args()
    try:
        run(args.input_dir,args.output_dir,limit_rows=args.limit_rows)
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        print(f"FAILED: {exc}\nInput files unchanged. Do not use an INCOMPLETE output.",file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
