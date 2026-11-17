"""SQLite 访问:每次操作独立连接,提交即持久化;WAL 模式支持读写并发。"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  run_number    TEXT NOT NULL UNIQUE,
  status        TEXT NOT NULL CHECK (status IN ('processing', 'published', 'failed')),
  error         TEXT,
  staging_dir   TEXT NOT NULL,
  r1_name       TEXT NOT NULL DEFAULT '',
  r2_name       TEXT NOT NULL DEFAULT '',
  i1_name       TEXT NOT NULL DEFAULT '',
  samples_name  TEXT NOT NULL DEFAULT '',
  total_pairs   INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  published_at  TEXT,
  label_revision INTEGER NOT NULL DEFAULT 0  -- 样本号更正代数,0 = 未更正
);
CREATE TABLE IF NOT EXISTS sample_counts (
  run_id     INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  sample_id  TEXT NOT NULL,                -- 拆样时的原始组名(存储名);'undetermined' 表示未分配组
  index_seq  TEXT NOT NULL DEFAULT '',
  pairs      INTEGER NOT NULL,
  corrected  INTEGER NOT NULL DEFAULT 0,   -- 其中经单碱基纠错的对数
  PRIMARY KEY (run_id, sample_id)
);
-- 当前对外样本号标签:磁盘文件始终以 storage_id 命名且不再改动,
-- 更正只改这里的 label;下载按 label -> storage_id 解析。
CREATE TABLE IF NOT EXISTS sample_labels (
  run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  storage_id  TEXT NOT NULL,               -- 发布目录里的文件前缀,不可变
  label       TEXT NOT NULL,               -- 当前对外样本号
  updated_at  TEXT NOT NULL,
  PRIMARY KEY (run_id, storage_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS sample_labels_label_key
  ON sample_labels(run_id, label);
-- 每次更正的审计记录
CREATE TABLE IF NOT EXISTS label_corrections (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id     INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  revision   INTEGER NOT NULL,             -- 更正后的标签版本号
  mapping    TEXT NOT NULL,                -- JSON: [[旧号, 新号], ...]
  created_at TEXT NOT NULL
);
"""


@contextmanager
def db_conn(path):
    """成功提交、异常回滚、用完关闭的连接上下文。"""
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path):
    with db_conn(path) as conn:
        conn.executescript(SCHEMA)
        conn.execute("PRAGMA journal_mode = WAL")
        # 旧库迁移:runs 表补 label_revision 列
        cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        if "label_revision" not in cols:
            conn.execute(
                "ALTER TABLE runs ADD COLUMN label_revision INTEGER NOT NULL DEFAULT 0"
            )
        # 旧库迁移:为已有计数但尚无标签映射的运行补建初始映射(标签=原始组名),幂等
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO sample_labels(run_id, storage_id, label, updated_at) "
            "SELECT sc.run_id, sc.sample_id, sc.sample_id, ? FROM sample_counts sc "
            "WHERE NOT EXISTS "
            "(SELECT 1 FROM sample_labels sl WHERE sl.run_id = sc.run_id)",
            (now,),
        )
