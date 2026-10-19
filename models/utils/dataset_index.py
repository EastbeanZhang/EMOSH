import json
import os
import sqlite3
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

INDEX_NAME = "dataset.sqlite3"


@contextmanager
def file_lock(path):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _row(record_id, split, info, signature=""):
    return (
        record_id,
        split,
        info.get("video_dir", "."),
        info["frames_number"],
        json.dumps(info, ensure_ascii=False, separators=(",", ":")),
        signature,
    )


class IndexWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / INDEX_NAME
        with file_lock(self.root / ".index.lock"):
            if not path.exists():
                with NamedTemporaryFile(dir=self.root, prefix=".index-", delete=False) as stream:
                    temporary = Path(stream.name)
                connection = sqlite3.connect(temporary)
                try:
                    with connection:
                        connection.execute(
                            "CREATE TABLE records (id TEXT PRIMARY KEY, split TEXT NOT NULL, "
                            "video_dir TEXT NOT NULL, frames_number INTEGER NOT NULL, "
                            "record_json TEXT NOT NULL, signature TEXT NOT NULL)"
                        )
                        connection.execute("CREATE INDEX records_split ON records(split, id)")
                        connection.execute(
                            "CREATE INDEX records_frames ON records(split, frames_number)"
                        )
                        connection.execute("PRAGMA user_version=1")
                        old = self.root / "videos_info.json"
                        if old.exists():
                            info = json.loads(old.read_text())
                            splits = json.loads((self.root / "videos_split.json").read_text())
                            membership = {}
                            for split, ids in splits.items():
                                for record_id in ids:
                                    if record_id in membership:
                                        raise ValueError(
                                            f"Record belongs to multiple splits: {record_id}"
                                        )
                                    membership[record_id] = split
                            unknown = membership.keys() - info.keys()
                            if unknown:
                                raise ValueError("Split index references missing records")
                            connection.executemany(
                                "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)",
                                (
                                    _row(key, membership.get(key, "test"), value)
                                    for key, value in info.items()
                                ),
                            )
                    connection.close()
                    os.replace(temporary, path)
                finally:
                    connection.close()
                    temporary.unlink(missing_ok=True)
        self.connection = sqlite3.connect(path, timeout=60)
        if self.connection.execute("PRAGMA user_version").fetchone()[0] != 1:
            self.connection.close()
            raise ValueError("Unsupported dataset index version")

    def directory(self, record_id):
        row = self.connection.execute(
            "SELECT video_dir FROM records WHERE id=?", (record_id,)
        ).fetchone()
        return row[0] if row else None

    def add(self, records):
        with self.connection:
            # Lock before checking IDs so independent packers cannot race on inserts.
            self.connection.execute("BEGIN IMMEDIATE")
            for record in records:
                values = _row(record["id"], record["split"], record["info"], record["signature"])
                existing = self.connection.execute(
                    "SELECT split, record_json, signature FROM records WHERE id=?", (record["id"],)
                ).fetchone()
                if existing:
                    if existing != (values[1], values[4], values[5]):
                        raise ValueError(f"Conflicting indexed record: {record['id']}")
                else:
                    self.connection.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)", values)

    def close(self):
        self.connection.close()


class DatasetIndex(Mapping):
    def __init__(self, root, cache_size=8):
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        self.root = Path(root)
        self.cache_size = cache_size
        self._connection = None
        self._pid = None
        self._cache = OrderedDict()
        self._legacy = None
        self._splits = None
        if not (self.root / INDEX_NAME).is_file():
            self._legacy = json.loads((self.root / "videos_info.json").read_text())
            splits = self.root / "videos_split.json"
            self._splits = json.loads(splits.read_text()) if splits.exists() else {}

    def _connect(self):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if self._connection is None:
            uri = (self.root / INDEX_NAME).resolve().as_uri() + "?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True, timeout=60)
            if self._connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("Unsupported dataset index version")
        return self._connection

    def __getitem__(self, key):
        if self._legacy is not None:
            return self._legacy[key]
        connection = self._connect()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        row = connection.execute("SELECT record_json FROM records WHERE id=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        info = json.loads(row[0])
        self._cache[key] = info
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return info

    def __contains__(self, key):
        if self._legacy is not None:
            return key in self._legacy
        return (
            self._connect().execute("SELECT 1 FROM records WHERE id=?", (key,)).fetchone()
            is not None
        )

    def iter_ids(self, split="test"):
        if self._legacy is not None:
            return iter(self._legacy if split == "test" else self._splits.get(split, []))
        query, parameters = "SELECT id FROM records", ()
        if split != "test":
            query += " WHERE split=?"
            parameters = (split,)
        return (row[0] for row in self._connect().execute(query + " ORDER BY id", parameters))

    def __iter__(self):
        return self.iter_ids()

    def __len__(self):
        if self._legacy is not None:
            return len(self._legacy)
        return self._connect().execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def summary(self, split):
        if self._legacy is not None:
            counts = [self._legacy[key]["frames_number"] for key in self.iter_ids(split)]
            return len(counts), sum(counts), min(counts, default=0)
        query, parameters = (
            "SELECT COUNT(*), COALESCE(SUM(frames_number), 0), COALESCE(MIN(frames_number), 0) FROM records",
            (),
        )
        if split != "test":
            query += " WHERE split=?"
            parameters = (split,)
        return self._connect().execute(query, parameters).fetchone()

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_connection=None, _pid=None, _cache=OrderedDict())
        return state

    def __del__(self):
        if hasattr(self, "_connection"):
            self.close()


def export_json(root):
    index = DatasetIndex(root)
    try:
        index._connect().execute("BEGIN")
        for filename, kind in (("videos_info.json", "records"), ("videos_split.json", "splits")):
            with NamedTemporaryFile(mode="w", dir=root, prefix=".export-", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    stream.write("{\n")
                    if kind == "records":
                        for number, key in enumerate(index):
                            if number:
                                stream.write(",\n")
                            stream.write(
                                json.dumps(key) + ":" + json.dumps(index[key], ensure_ascii=False)
                            )
                    else:
                        for number, split in enumerate(("train", "valid")):
                            if number:
                                stream.write(",\n")
                            stream.write(json.dumps(split) + ":[")
                            for offset, key in enumerate(index.iter_ids(split)):
                                if offset:
                                    stream.write(",")
                                stream.write(json.dumps(key))
                            stream.write("]")
                    stream.write("\n}\n")
                    stream.close()
                    os.replace(temporary, Path(root) / filename)
                finally:
                    temporary.unlink(missing_ok=True)
    finally:
        index.close()
