"""API contract and snapshot replacement tests."""
from contextlib import closing
import json
import errno
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from build_snapshot import SCHEMA
from serve_snapshot import create_app, resolve_snapshot, with_static_files


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / "snapshot.sqlite"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executescript(SCHEMA)
            db.execute("INSERT INTO metadata VALUES ('schema_version','2')")
            db.execute("INSERT INTO countries VALUES ('RU','vpn','ensemble','{}',1)")
            db.executemany("INSERT INTO series VALUES ('RU',?,?)", [('2025-01-01', 1), ('2025-01-02', 10)])
            db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       ('RU', '2025-01-01', '2025-01-02', '2025-01-02', 20, 2, 0,
                        None, None, 'Example event', 'Example', 'filtering', 'protest',
                        'security', '["VPN"]', '["https://example.org"]', '{}'))
        self.app = create_app(self.database)

    def request(self, path, query='', method='GET', app=None):
        response = []
        body = b''.join((app or self.app)({'PATH_INFO': path, 'QUERY_STRING': query,
                                         'REQUEST_METHOD': method}, lambda s, h: response.append((s, h))))
        return response[0][0], json.loads(body)

    def test_contract(self):
        self.assertEqual(self.request('/api/health')[0], '200 OK')
        self.assertEqual(self.request('/api/countries')[1][0]['end'], '2025-01-02')
        self.assertEqual(len(self.request('/api/series', 'country=RU&start=2025-01-02')[1]), 1)
        event = self.request('/api/events', 'country=RU&start=2025-01-02')[1][0]
        self.assertEqual(event['affectedServices'], ['VPN'])
        self.assertFalse(event['labelMatched'])
        self.assertEqual(event['cause'], 'protest')
        self.assertEqual(self.request('/api/series', 'country=RU&limit=1&offset=1')[1][0]['rate'], 10)

    def test_validation(self):
        for query in ('country=ru', 'country=RU&country=US', 'country=RU&start=2025-02-30',
                      'country=RU&limit=0', 'country=RU&offset=-1', 'country=RU&limit=no',
                      'country=RU&start=2025-02-02&end=2025-01-01', 'unexpected=x'):
            self.assertEqual(self.request('/api/series', query)[0], '400 Bad Request', query)
        self.assertEqual(self.request('/api/series')[0], '400 Bad Request')
        self.assertEqual(self.request('/api/series', 'country=ZZ')[1], [])
        self.assertEqual(self.request('/api/events', method='POST')[0], '405 Method Not Allowed')
        self.assertEqual(self.request('/missing')[0], '404 Not Found')

    def test_event_filters(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("INSERT INTO labels VALUES (0, 'example', '{}')")
            db.execute("INSERT INTO countries VALUES ('ID','vpn','ensemble','{}',1)")
            db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       ('ID', '2025-02-01', '2025-02-02', '2025-02-02', 30, 3, 0,
                        0, 1, 'Restriction', 'Example', 'filtering', 'Protests',
                        'public_safety', '["Facebook", "Instagram"]', '[]', '{}'))
        self.assertEqual(len(self.request('/api/events')[1]), 2)
        for filters in (
            {'labelMatched': 'true'}, {'affectedServices': 'facebook'},
            {'cause': 'protests'}, {'justification': 'PUBLIC_SAFETY'},
            {'category': 'filtering', 'affectedServices': 'Instagram', 'country': 'ID',
             'labelMatched': 'true', 'start': '2025-02-02', 'limit': 1},
        ):
            status, events = self.request('/api/events', urlencode(filters))
            self.assertEqual(status, '200 OK')
            self.assertEqual([e['country'] for e in events], ['ID'], filters)
        self.assertEqual(self.request('/api/events', 'labelMatched=false')[1][0]['country'], 'RU')
        self.assertEqual(self.request('/api/events', 'category=filtering&limit=1&offset=1')[1][0]['country'], 'RU')
        for query in ('affectedServices=Face', 'cause=unknown', 'country=RU&affectedServices=Facebook',
                      'labelMatched=true&offset=1', urlencode({'category': "' OR 1=1 --"})):
            self.assertEqual(self.request('/api/events', query)[1], [], query)
        for query in ('labelMatched=yes', 'labelMatched=', 'category=', 'cause=+',
                      'justification=', 'affectedServices=', 'category=a&category=b'):
            self.assertEqual(self.request('/api/events', query)[0], '400 Bad Request', query)
        self.assertEqual(self.request('/api/series', 'country=RU&labelMatched=true')[0], '400 Bad Request')

    def test_current_symlink_refresh(self):
        current = self.root / 'current.sqlite'
        current.symlink_to(self.database.name)
        app = create_app(current)
        self.assertEqual(len(self.request('/api/countries', app=app)[1]), 1)
        other = self.root / 'other.sqlite'
        with closing(sqlite3.connect(other)) as db, db:
            db.executescript(SCHEMA)
            db.execute("INSERT INTO metadata VALUES ('schema_version','2')")
        replacement = self.root / 'replacement'
        replacement.symlink_to(other.name)
        os.replace(replacement, current)
        self.assertEqual(self.request('/api/countries', app=app)[1], [])

    def test_missing_database_does_not_create_file(self):
        missing = self.root / 'missing.sqlite'
        with self.assertLogs(level='ERROR'):
            self.assertEqual(self.request('/api/health', app=create_app(missing))[0], '503 Service Unavailable')
        self.assertFalse(missing.exists())

    def test_shared_symlink_cache_refresh(self):
        with patch.object(Path, 'resolve', side_effect=[OSError(errno.EINVAL, 'stale symlink'), self.database]), \
                patch('serve_snapshot.os.scandir', wraps=os.scandir) as scan:
            self.assertEqual(resolve_snapshot(self.root / 'current' / 'snapshot.sqlite'), self.database)
            scan.assert_called_once_with(self.root)

    def test_static_root_isolation(self):
        public = self.root / 'public'
        public.mkdir()
        app = with_static_files(self.app, public)
        status = []
        body = b''.join(app({'PATH_INFO': '/../snapshot.sqlite', 'REQUEST_METHOD': 'GET'},
                            lambda s, h: status.append(s)))
        self.assertEqual(status, ['404 Not Found'])
        self.assertEqual(body, b'Not found')


if __name__ == '__main__':
    unittest.main()
