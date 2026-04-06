#!/usr/bin/env python3
"""
load_explanations.py

Usage:
    python3 load_explanations.py <csv_file> <db_name> <db_user> <socket_dir>

The CSV must have columns:
    summary, links, tags, cause, affected_regions, scope, startdate

• Array columns (`links`, `tags`, `affected_regions`) can be:
      a) JSON list, e.g. ["https://a.com","https://b.org"]
      b) Semicolon-separated text,  e.g. https://a.com; https://b.org
• scope must be national / regional (loader lower-cases for safety)
• startdate: YYYY-MM-DD
"""
import sys, os, json, csv, psycopg2
from datetime import datetime

def parse_array(raw):
    """Accept JSON list or semi-colon separated string → Python list / None."""
    if raw is None or str(raw).strip() == "" or str(raw).upper() == "NULL":
        return None
    raw = raw.strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [item.strip() for item in raw.split(";") if item.strip()]

def parse_scope(raw):
    return raw.lower() if raw and raw.lower() in ("national", "regional") else None

def parse_date(raw):
    return datetime.strptime(raw, "%Y-%m-%d").date() if raw else None

def row_to_tuple(r):
    """Return values in the DB column order (except id)."""
    return (
        r["summary"].strip(),
        parse_array(r.get("links")),
        parse_array(r.get("tags")),
        (r.get("cause") or None),
        parse_array(r.get("affected_regions")),
        parse_scope(r.get("scope")),
        parse_date(r.get("startdate"))
    )

def load_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            yield row_to_tuple(r)

def upsert(cur, rows):
    """
    One execute_values() batch → INSERT … ON CONFLICT …
    Updates every column if the row already exists.
    """
    from psycopg2.extras import execute_values
    sql = """
        INSERT INTO explanations
               (summary, links, tags, cause, affected_regions, scope, startdate)
        VALUES %s
        ON CONFLICT (summary, startdate) DO UPDATE
        SET   links            = EXCLUDED.links,
              tags             = EXCLUDED.tags,
              cause            = EXCLUDED.cause,
              affected_regions = EXCLUDED.affected_regions,
              scope            = EXCLUDED.scope;
    """
    execute_values(cur, sql, rows)

def main(csv_file, db_name, db_user, socket_dir):
    conn = psycopg2.connect(dbname=db_name,
                            user=db_user,
                            host=os.path.abspath(socket_dir))
    cur = conn.cursor()

    rows = list(load_csv(csv_file))
    if not rows:
        print("[!] CSV is empty – nothing to do")
        return

    upsert(cur, rows)
    conn.commit()
    cur.close()
    conn.close()
    print(f"[✓] Up-serted {len(rows)} explanations from {csv_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 load_explanations.py <csv_file> <db_name> <db_user> <socket_dir>")
        sys.exit(1)
    main(*sys.argv[1:])
