#!/usr/bin/env python3
"""Read-only HTTP API for atomically published CenAlert snapshots (WSGI)."""

import argparse
from contextlib import closing
from datetime import date
import errno
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server


def resolve_snapshot(database):
    try:
        return database.resolve(strict=True)
    except OSError as error:
        if error.errno != errno.EINVAL:
            raise
        # Docker Desktop may retain the old symlink inode after a host rename.
        # Reading its containing directory refreshes the shared directory entries.
        with os.scandir(database.parent.parent) as entries:
            list(entries)
        return database.resolve(strict=True)


def create_app(database):
    # Resolve on each request so run_daily's current symlink can advance.
    database = Path(database).absolute()

    def app(environ, start_response):
        status = "200 OK"
        try:
            if environ.get("REQUEST_METHOD") != "GET":
                status, result = "405 Method Not Allowed", {"error": "Use GET"}
            else:
                query = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
                if any(len(v) != 1 for v in query.values()):
                    raise ValueError("Repeated query parameter")
                args = {k: v[0] for k, v in query.items()}
                path = environ.get("PATH_INFO", "")
                if path not in ("/api/health", "/api/countries", "/api/series", "/api/events"):
                    status, result = "404 Not Found", {"error": "Unknown endpoint"}
                else:
                    with closing(sqlite3.connect(resolve_snapshot(database).as_uri() + "?mode=ro", uri=True)) as conn:
                        conn.row_factory = sqlite3.Row
                        conn.execute("PRAGMA query_only=ON")
                        conn.execute("BEGIN")
                        metadata = dict(conn.execute("SELECT key,value FROM metadata"))
                        if metadata.get("schema_version") != "2":
                            raise sqlite3.DatabaseError("Unsupported snapshot schema")
                        result = read_endpoint(conn, path, args, metadata)
                        conn.rollback()
        except ValueError as error:
            status, result = "400 Bad Request", {"error": str(error)}
        except (sqlite3.Error, OSError):
            logging.exception("Snapshot unavailable")
            status, result = "503 Service Unavailable", {"error": "Snapshot unavailable"}
        body = json.dumps(result, allow_nan=False).encode()
        headers = [("Content-Type", "application/json; charset=utf-8"),
                   ("Content-Length", str(len(body))), ("Cache-Control", "no-store"),
                   ("X-Content-Type-Options", "nosniff")]
        if status.startswith("405"):
            headers.append(("Allow", "GET"))
        start_response(status, headers)
        return [body]
    return app


def read_endpoint(conn, path, args, metadata):
    allowed = {"country", "start", "end", "limit", "offset"} if path in ("/api/series", "/api/events") else set()
    if path == "/api/events":
        allowed |= {"labelMatched", "category", "cause", "justification", "affectedServices"}
    if set(args) - allowed:
        raise ValueError("Unknown query parameter")
    if path == "/api/health":
        return {"status": "ok", "created_at": metadata.get("created_at"), "schema_version": "2"}
    if path == "/api/countries":
        return [dict(r) for r in conn.execute(
            "SELECT c.country,c.topic,c.algorithm,MIN(s.date) AS start,MAX(s.date) AS end "
            "FROM countries c LEFT JOIN series s USING(country) GROUP BY c.country ORDER BY c.country")]
    country = args.get("country")
    if country is not None and not re.fullmatch(r"[A-Z]{2}", country):
        raise ValueError("country must be a two-letter uppercase code")
    if path == "/api/series" and not country:
        raise ValueError("country is required")
    for key in ("start", "end"):
        if key in args and (not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args[key]) or date.fromisoformat(args[key]).isoformat() != args[key]):
            raise ValueError(f"{key} must be YYYY-MM-DD")
    if args.get("start", "") > args.get("end", "9999-12-31"):
        raise ValueError("start must not exceed end")
    limit, offset = int(args.get("limit", 20000)), int(args.get("offset", 0))
    if not 1 <= limit <= 20000 or offset < 0:
        raise ValueError("limit must be 1–20000 and offset must be nonnegative")
    where, values = [], []
    if country:
        where.append("country=?")
        values.append(country)
    series = path == "/api/series"
    if not series:
        if "labelMatched" in args:
            if args["labelMatched"] not in ("true", "false"):
                raise ValueError("labelMatched must be true or false")
            where.append("label_index IS NOT NULL" if args["labelMatched"] == "true" else "label_index IS NULL")
        for field in ("category", "cause", "justification", "affectedServices"):
            if field not in args:
                continue
            value = args[field].strip()
            if not value:
                raise ValueError(f"{field} must not be empty")
            if field == "affectedServices":
                where.append("EXISTS (SELECT 1 FROM json_each(events.affected_services) WHERE value = ? COLLATE NOCASE)")
            else:
                where.append(f"{field} = ? COLLATE NOCASE")
            values.append(value)
    for key, column, op in (("start", "date" if series else "end", ">="),
                            ("end", "date" if series else "start", "<=")):
        if key in args:
            where.append(f"{column}{op}?")
            values.append(args[key])
    clause = " WHERE " + " AND ".join(where) if where else ""
    table, order = ("series", "date") if series else ("events", "start DESC,country")
    rows = conn.execute(f"SELECT * FROM {table}{clause} ORDER BY {order} LIMIT ? OFFSET ?", (*values, limit, offset))
    if series:
        return [dict(country=r["country"], date=r["date"], rate=r["value"]) for r in rows]
    result = []
    for row in rows:
        event = {k: row[k] for k in ("country", "peak", "impact", "title", "description", "category", "cause", "justification")}
        event.update(startDate=row["start"], endDate=row["end"],
                     affectedServices=json.loads(row["affected_services"]), sources=json.loads(row["sources"]),
                     labelMatched=row["label_index"] is not None, matchDistanceDays=row["proximity"])
        result.append(event)
    return result


application = create_app(os.environ.get("CENALERT_DATABASE", "/data/current/cenalert.sqlite"))


def with_static_files(api, directory):
    """Serve the built dashboard for local development; production uses nginx."""
    root = Path(directory).resolve()

    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        if path.startswith("/api/"):
            return api(environ, start_response)
        target = (root / (path.lstrip("/") or "index.html")).resolve()
        if environ.get("REQUEST_METHOD") != "GET" or not target.is_relative_to(root) or not target.is_file():
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not found"]
        body = target.read_bytes()
        start_response("200 OK", [("Content-Type", mimetypes.guess_type(target)[0] or "application/octet-stream"),
                                  ("Content-Length", str(len(body))), ("X-Content-Type-Options", "nosniff")])
        return [body]
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--static-directory", type=Path, help="Built dashboard directory for local preview")
    args = parser.parse_args()
    app = create_app(args.database)
    if args.static_directory:
        app = with_static_files(app, args.static_directory)
    with make_server(args.host, args.port, app) as server:
        print(f"CenAlert API: http://{args.host}:{args.port}", flush=True)
        server.serve_forever()
