"""关键情形自测:拆分正确性、纠错规则、损坏处理、原子发布、恢复、不可改写、样本号更正。"""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.demux import (  # noqa: E402
    FastqFormatError,
    assign_index,
    build_matcher,
    iter_fastq,
    read_id,
)
from app.samples import SampleSheetError, load_samples  # noqa: E402

SAMPLES_CSV = "sample_id,index\nS1,AAAACCCC\nS2,AAAACCGG\nS3,GGGGTTTT\n"

# S1=AAAACCCC 与 S2=AAAACCGG 相距 2,便于构造歧义错配;S3 远离二者
READS = [
    ("r001", "AAAACCCC"),  # S1 精确
    ("r002", "AAAACCGG"),  # S2 精确
    ("r003", "AAAACCCA"),  # 距 S1 唯一单碱基错配 -> 纠错到 S1
    ("r004", "AAAACCGC"),  # 距 S1、S2 各 1 -> 歧义 -> 未分配
    ("r005", "NNNNNNNN"),  # 无候选 -> 未分配
    ("r006", "GGGGTTTA"),  # 距 S3 唯一单碱基错配 -> 纠错到 S3
    ("r007", "AAAACCCC"),  # S1 精确(检验同组内顺序)
]

SEQ1, SEQ2 = "ACGT" * 10, "TGCA" * 10
QUAL = "I" * 40


def fq(records):
    return "".join(f"@{h}\n{s}\n+\n{q}\n" for h, s, q in records)


def make_inputs(reads=READS):
    r1, r2, i1 = [], [], []
    for n, (rid, idx) in enumerate(reads):
        head = f"INST:1:FC:1:1101:{1000 + n}:{2000 + n}"
        r1.append((f"{head} 1:N:0:0", SEQ1, QUAL))
        r2.append((f"{head} 2:N:0:0", SEQ2, QUAL))
        i1.append((f"{head} 1:N:0:0", idx, "I" * len(idx)))
    return fq(r1).encode(), fq(r2).encode(), fq(i1).encode(), SAMPLES_CSV.encode()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        c.data_dir = tmp_path / "data"
        yield c


def submit(client, run_number, r1, r2, i1, samples):
    return client.post(
        "/runs",
        data={"run_number": run_number},
        files={
            "r1": ("r1.fastq", r1),
            "r2": ("r2.fastq", r2),
            "i1": ("i1.fastq", i1),
            "samples": ("samples.csv", samples),
        },
        follow_redirects=False,
    )


def submit_good(client, run_number="RUN1"):
    r1, r2, i1, csv = make_inputs()
    return submit(client, run_number, r1, r2, i1, csv)


def run_id_of(resp):
    assert resp.status_code == 303, resp.text
    return int(resp.headers["location"].rsplit("/", 1)[1])


def wait_done(client, run_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/runs/{run_id}").json()
        if st["status"] != "processing":
            return st
        time.sleep(0.05)
    raise TimeoutError("处理超时")


def parse_fastq_ids(text):
    lines = text.splitlines()
    return [lines[i][1:].split()[0] for i in range(0, len(lines), 4)]


def head_of(n):
    """第 n 条 READS 记录对应的读段号(见 make_inputs)。"""
    return f"INST:1:FC:1:1101:{1000 + n}:{2000 + n}"


def counts_of(status):
    return {c["sample_id"]: (c["pairs"], c["corrected"]) for c in status["counts"]}


# ------------------------------------------------------------ 单元级

def test_assign_index_rules():
    samples = [("S1", "AAAACCCC"), ("S2", "AAAACCGG"), ("S3", "GGGGTTTT")]
    exact, neighbors = build_matcher(samples)
    assert assign_index("AAAACCCC", exact, neighbors) == ("S1", False)   # 精确优先
    assert assign_index("AAAACCCA", exact, neighbors) == ("S1", True)    # 唯一错配
    assert assign_index("AAAACCGC", exact, neighbors) == (None, False)   # 歧义
    assert assign_index("NNNNNNNN", exact, neighbors) == (None, False)   # 无候选
    assert assign_index("AAAA", exact, neighbors) == (None, False)       # 长度不符


def test_read_id():
    assert read_id("INST:1:FC:1:1101:100:100 1:N:0:0") == "INST:1:FC:1:1101:100:100"
    assert read_id("PLAINID") == "PLAINID"
    assert read_id("") == ""


def test_iter_fastq_truncated(tmp_path):
    p = tmp_path / "bad.fastq"
    p.write_text("@a\nACGT\n+\n")  # 缺质量行
    with pytest.raises(FastqFormatError, match="截断"):
        list(iter_fastq(str(p), "R1"))


def test_iter_fastq_len_mismatch(tmp_path):
    p = tmp_path / "bad.fastq"
    p.write_text("@a\nACGT\n+\nIII\n")  # 质量短 1
    with pytest.raises(FastqFormatError, match="长度"):
        list(iter_fastq(str(p), "R1"))


def test_samples_validation(tmp_path):
    def load(text):
        p = tmp_path / "s.csv"
        p.write_text(text)
        return load_samples(str(p))

    assert load("sid,idx\nA,ACGT\n") == [("A", "ACGT")]
    assert load("A,acgt\n") == [("A", "ACGT")]  # 小写归一、无表头
    with pytest.raises(SampleSheetError, match="重复"):
        load("A,ACGT\nB,ACGT\n")
    with pytest.raises(SampleSheetError, match="长度不一致"):
        load("A,ACGT\nB,ACGTA\n")
    with pytest.raises(SampleSheetError, match="非法字符"):
        load("A,ACGT\nB,ACGB\n")
    with pytest.raises(SampleSheetError, match="保留字"):
        load("undetermined,ACGT\n")


# ------------------------------------------------------------ 端到端

def test_happy_path(client):
    rid = run_id_of(submit_good(client))
    st = wait_done(client, rid)
    assert st["status"] == "published", st["error"]
    assert st["total_pairs"] == 7
    assert counts_of(st) == {
        "S1": (3, 1), "S2": (1, 0), "S3": (1, 1), "undetermined": (2, 0),
    }

    s1_r1 = client.get(f"/runs/{rid}/download/S1_R1.fastq")
    assert s1_r1.status_code == 200
    assert parse_fastq_ids(s1_r1.text) == [head_of(0), head_of(2), head_of(6)]  # 原顺序
    s1_r2 = client.get(f"/runs/{rid}/download/S1_R2.fastq")
    assert parse_fastq_ids(s1_r2.text) == [head_of(0), head_of(2), head_of(6)]
    assert SEQ2 in s1_r2.text  # R2 内容确实是 R2
    und = client.get(f"/runs/{rid}/download/undetermined_R1.fastq")
    assert parse_fastq_ids(und.text) == [head_of(3), head_of(4)]

    log = client.get(f"/runs/{rid}/download/demux.log")
    assert log.status_code == 200 and "总读段对: 7" in log.text

    page = client.get(f"/runs/{rid}")
    assert page.status_code == 200 and "AAAACCCC" in page.text and "已发布" in page.text


def test_truncated_input_fails_then_retry_succeeds(client):
    r1, r2, i1, csv = make_inputs()
    bad_r2 = r2[: r2.rfind(b"\n", 0, r2.rfind(b"\n"))]  # 末条记录缺质量行
    rid = run_id_of(submit(client, "RUN2", r1, bad_r2, i1, csv))
    st = wait_done(client, rid)
    assert st["status"] == "failed"
    assert "截断" in st["error"]
    # 不发布半套文件
    assert not (client.data_dir / "published" / "RUN2").exists()
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 409

    # 同一运行号重试可完成
    rid2 = run_id_of(submit_good(client, "RUN2"))
    st2 = wait_done(client, rid2)
    assert st2["status"] == "published"
    assert st2["total_pairs"] == 7
    assert client.get(f"/runs/{rid2}/download/S1_R1.fastq").status_code == 200


def test_read_id_mismatch_fails(client):
    r1, r2, i1, csv = make_inputs()
    bad_r2 = r2.replace(b"1101:1001:2001", b"1101:9999:2001", 1)
    rid = run_id_of(submit(client, "RUN3", r1, bad_r2, i1, csv))
    st = wait_done(client, rid)
    assert st["status"] == "failed" and "读段号不一致" in st["error"]
    assert not (client.data_dir / "published" / "RUN3").exists()


def test_seq_qual_len_mismatch_fails(client):
    r1, r2, i1, csv = make_inputs()
    bad_i1 = i1.replace(b"IIIIIIII\n", b"IIIIIII\n", 1)  # 首条 I1 质量少 1
    rid = run_id_of(submit(client, "RUN4", r1, r2, bad_i1, csv))
    st = wait_done(client, rid)
    assert st["status"] == "failed" and "长度" in st["error"]


def test_record_count_mismatch_fails(client):
    r1, r2, i1, csv = make_inputs()
    extra = f"@INST:1:FC:1:1101:9:9 1:N:0:0\n{SEQ1}\n+\n{QUAL}\n".encode()
    rid = run_id_of(submit(client, "RUN5", r1 + extra, r2, i1, csv))
    st = wait_done(client, rid)
    assert st["status"] == "failed" and "记录数不一致" in st["error"]


def test_published_run_immutable(client):
    rid = run_id_of(submit_good(client, "RUN6"))
    assert wait_done(client, rid)["status"] == "published"
    before = client.get(f"/runs/{rid}/download/S1_R1.fastq").text

    resp = submit_good(client, "RUN6")  # 同号再提交
    assert resp.status_code == 409
    assert "不可改写" in resp.json()["detail"]
    after = client.get(f"/runs/{rid}/download/S1_R1.fastq").text
    assert before == after  # 文件未被改写


def test_bad_sample_sheet_rejected(client):
    r1, r2, i1, _ = make_inputs()
    dup = b"A,ACGT\nB,ACGT\n"
    resp = submit(client, "RUN7", r1, r2, i1, dup)
    assert resp.status_code == 400
    assert "重复" in resp.json()["detail"]


def test_interrupted_run_recovered_on_startup(client):
    """模拟进程中断:遗留 processing 记录 + 半成品目录,重启后标失败、清理干净。"""
    from app.main import recover_interrupted
    from app.db import db_conn
    from app.config import db_path, staging_dir, published_dir

    staging = staging_dir() / "deadbeef"
    (staging / "out").mkdir(parents=True)
    (staging / "out" / "S1_R1.fastq").write_text("半成品")
    stray_pub = published_dir() / "RUN8"
    stray_pub.mkdir(parents=True)
    (stray_pub / "S1_R1.fastq").write_text("改名后来不及落库的文件")
    with db_conn(db_path()) as conn:
        conn.execute(
            "INSERT INTO runs(run_number, status, staging_dir, created_at, updated_at) "
            "VALUES ('RUN8', 'processing', ?, 't', 't')",
            (str(staging),),
        )

    recover_interrupted()  # 模拟重启

    with db_conn(db_path()) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_number='RUN8'").fetchone()
    assert row["status"] == "failed" and "中断" in row["error"]
    assert not staging.exists()
    assert not stray_pub.exists()
    # 恢复后同一运行号可重试并完成
    rid = run_id_of(submit_good(client, "RUN8"))
    assert wait_done(client, rid)["status"] == "published"


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "新建拆样运行" in resp.text


# ------------------------------------------------------------ 样本号更正

def relabel(client, run_id, mapping_text):
    return client.post(
        f"/runs/{run_id}/relabel",
        data={"mapping": mapping_text},
        follow_redirects=False,
    )


def published_run(client, run_number):
    rid = run_id_of(submit_good(client, run_number))
    st = wait_done(client, rid)
    assert st["status"] == "published", st["error"]
    return rid


def test_relabel_basic(client):
    """单组改名:计数/索引/双端下载按新号对应原读段,旧地址失效,内容与对数不变。"""
    rid = published_run(client, "RUNC1")
    s1_r1 = client.get(f"/runs/{rid}/download/S1_R1.fastq").text
    s1_r2 = client.get(f"/runs/{rid}/download/S1_R2.fastq").text

    resp = relabel(client, rid, "S1,POOL-A")
    assert resp.status_code == 303, resp.text

    st = client.get(f"/api/runs/{rid}").json()
    assert st["label_revision"] == 1
    counts = counts_of(st)
    assert counts["POOL-A"] == (3, 1)   # 计数跟随原读段
    assert "S1" not in counts
    assert counts["S2"] == (1, 0)       # 未更正组不受影响
    idx = {c["sample_id"]: c["index_seq"] for c in st["counts"]}
    assert idx["POOL-A"] == "AAAACCCC"  # 索引序列跟随原组

    # 旧下载地址一律失效
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 410
    assert client.get(f"/runs/{rid}/download/v0/S1_R1.fastq").status_code == 410
    assert client.get(f"/runs/{rid}/download/demux.log").status_code == 410
    # 新地址按新号对应原读段,内容与对数不变
    r1 = client.get(f"/runs/{rid}/download/v1/POOL-A_R1.fastq")
    assert r1.status_code == 200 and r1.text == s1_r1
    r2 = client.get(f"/runs/{rid}/download/v1/POOL-A_R2.fastq")
    assert r2.status_code == 200 and r2.text == s1_r2
    assert parse_fastq_ids(r1.text) == [head_of(0), head_of(2), head_of(6)]
    assert client.get(f"/runs/{rid}/download/v1/demux.log").status_code == 200
    assert client.get(f"/runs/{rid}/download/v1/S2_R1.fastq").status_code == 200
    # 不存在的标签/超前版本
    assert client.get(f"/runs/{rid}/download/v1/S1_R1.fastq").status_code == 404
    assert client.get(f"/runs/{rid}/download/v2/POOL-A_R1.fastq").status_code == 404

    page = client.get(f"/runs/{rid}").text
    assert "POOL-A" in page and f"/runs/{rid}/download/v1/POOL-A_R1.fastq" in page
    assert f'href="/runs/{rid}/download/S1_R1.fastq"' not in page  # 页面不再给旧链接
    assert "S1 → POOL-A" in page  # 更正历史


def test_relabel_swap(client):
    """两组互换:新号对应原来的读段;旧地址失效,绝不会在互换后取到另一组读段。"""
    rid = published_run(client, "RUNC2")
    s1_r1 = client.get(f"/runs/{rid}/download/S1_R1.fastq").text
    s2_r1 = client.get(f"/runs/{rid}/download/S2_R1.fastq").text
    s1_r2 = client.get(f"/runs/{rid}/download/S1_R2.fastq").text
    s2_r2 = client.get(f"/runs/{rid}/download/S2_R2.fastq").text
    assert s1_r1 != s2_r1

    resp = relabel(client, rid, "S1,S2\nS2,S1")  # 一次提交,两组互换
    assert resp.status_code == 303, resp.text

    st = client.get(f"/api/runs/{rid}").json()
    assert st["label_revision"] == 1
    counts = counts_of(st)
    assert counts["S1"] == (1, 0)   # 新 S1 = 原 S2 的对数
    assert counts["S2"] == (3, 1)   # 新 S2 = 原 S1 的对数

    # 旧地址一律失效:互换后绝不能从旧地址取到另一组的读段
    for name in ("S1_R1.fastq", "S2_R1.fastq", "S1_R2.fastq", "S2_R2.fastq", "demux.log"):
        assert client.get(f"/runs/{rid}/download/{name}").status_code == 410
        assert client.get(f"/runs/{rid}/download/v0/{name}").status_code == 410
    # 新地址:新号对应原来的读段(互换语义)
    assert client.get(f"/runs/{rid}/download/v1/S1_R1.fastq").text == s2_r1
    assert client.get(f"/runs/{rid}/download/v1/S2_R1.fastq").text == s1_r1
    assert client.get(f"/runs/{rid}/download/v1/S1_R2.fastq").text == s2_r2
    assert client.get(f"/runs/{rid}/download/v1/S2_R2.fastq").text == s1_r2


def test_relabel_chain_old_versions_invalid(client):
    """连续更正:每次更正后上一版地址立即失效;让出的名字可再用。"""
    rid = published_run(client, "RUNC3")
    orig_s1 = client.get(f"/runs/{rid}/download/S1_R1.fastq").text
    orig_s2 = client.get(f"/runs/{rid}/download/S2_R1.fastq").text

    assert relabel(client, rid, "S1,TMP").status_code == 303
    assert client.get(f"/runs/{rid}/download/v1/TMP_R1.fastq").text == orig_s1

    assert relabel(client, rid, "TMP,FINAL").status_code == 303
    st = client.get(f"/api/runs/{rid}").json()
    assert st["label_revision"] == 2
    assert client.get(f"/runs/{rid}/download/v1/TMP_R1.fastq").status_code == 410
    assert client.get(f"/runs/{rid}/download/v2/FINAL_R1.fastq").text == orig_s1
    # 已让出的名字不再是有效标签
    assert client.get(f"/runs/{rid}/download/v2/TMP_R1.fastq").status_code == 404
    assert client.get(f"/runs/{rid}/download/v2/S1_R1.fastq").status_code == 404

    # 让出的名字可被其他组再用(新 S1 = 原 S2 的读段)
    assert relabel(client, rid, "S2,S1").status_code == 303
    assert client.get(f"/runs/{rid}/download/v3/S1_R1.fastq").text == orig_s2
    assert client.get(f"/runs/{rid}/download/v2/FINAL_R1.fastq").status_code == 410


@pytest.mark.parametrize("mapping,msg", [
    ("S1,S2", "未参与本次更正"),          # 新号占用未更正组
    ("S1,undetermined", "保留字"),        # 占用未分配组名
    ("undetermined,X1", "不可改名"),      # 未分配组不可改名
    ("S9,X1", "不是当前有效样本号"),      # 旧号不存在
    ("S1,X1\nS1,X2", "旧号重复"),
    ("S1,X1\nS2,X1", "新号重复"),
    ("S1,S1", "相同"),
    ("S1,?bad", "无效"),                  # 新号格式非法
    ("S1", "格式应为"),                   # 行格式不对
    ("", "映射为空"),
    ("# 只有注释行\n", "映射为空"),
])
def test_relabel_validation(client, mapping, msg):
    rid = published_run(client, "RUNC4")
    before = client.get(f"/runs/{rid}/download/S1_R1.fastq").text
    resp = relabel(client, rid, mapping)
    assert resp.status_code == 400, resp.text
    assert msg in resp.json()["detail"]
    # 校验失败不产生任何变化:版本、计数、下载全部保持
    st = client.get(f"/api/runs/{rid}").json()
    assert st["label_revision"] == 0
    assert counts_of(st)["S1"] == (3, 1)
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").text == before


def test_relabel_requires_published(client):
    """未发布(处理失败)的运行不能更正。"""
    r1, r2, i1, csv = make_inputs()
    bad_r2 = r2[: r2.rfind(b"\n", 0, r2.rfind(b"\n"))]  # 末条记录缺质量行
    rid = run_id_of(submit(client, "RUNC5", r1, bad_r2, i1, csv))
    assert wait_done(client, rid)["status"] == "failed"
    resp = relabel(client, rid, "S1,X1")
    assert resp.status_code == 409


def test_relabel_write_failure_keeps_old_version(client):
    """更正事务写入失败:整体回滚,旧版标签与旧下载地址继续有效。"""
    import sqlite3 as sqlite

    rid = published_run(client, "RUNC6")
    before = client.get(f"/runs/{rid}/download/S1_R1.fastq").text

    # 注入故障:删掉审计表,让更正事务在末步写入失败
    raw = sqlite.connect(str(client.data_dir / "app.db"))
    raw.execute("DROP TABLE label_corrections")
    raw.commit()
    raw.close()

    with pytest.raises(Exception):  # 事务中途失败 -> 500,整体回滚
        relabel(client, rid, "S1,POOL-A")

    # 旧版完整保留:版本未变、旧地址仍供数、内容与计数不变
    st = client.get(f"/api/runs/{rid}").json()
    assert st["label_revision"] == 0
    assert counts_of(st)["S1"] == (3, 1)
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").text == before

    # 故障修复后同一更正可重试并成功
    from app.config import db_path
    from app.db import init_db
    init_db(db_path())
    assert relabel(client, rid, "S1,POOL-A").status_code == 303
    assert client.get(f"/runs/{rid}/download/v1/POOL-A_R1.fastq").text == before
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 410


def test_relabel_survives_restart(client):
    """重启(中断恢复)后更正仍然有效。"""
    rid = published_run(client, "RUNC7")
    orig = client.get(f"/runs/{rid}/download/S1_R1.fastq").text
    assert relabel(client, rid, "S1,POOL-A").status_code == 303

    from app.main import recover_interrupted
    recover_interrupted()  # 模拟进程重启

    st = client.get(f"/api/runs/{rid}").json()
    assert st["status"] == "published" and st["label_revision"] == 1
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 410
    assert client.get(f"/runs/{rid}/download/v1/POOL-A_R1.fastq").text == orig


def test_relabel_download_interleaving(client):
    """更正与下载交错:每次下载只能看到完整的一版标签,内容与版本严格对应。"""
    rid = published_run(client, "RUNC8")
    orig = {
        sid: client.get(f"/runs/{rid}/download/{sid}_R1.fastq").text
        for sid in ("S1", "S2")
    }
    assert orig["S1"] != orig["S2"]

    from fastapi.testclient import TestClient
    from app.main import app

    stop = threading.Event()
    errors = []

    def swapper():  # 反复互换 S1/S2:每次更正版本 +1
        c = TestClient(app)
        while not stop.is_set():
            c.post(f"/runs/{rid}/relabel", data={"mapping": "S1,S2\nS2,S1"},
                   follow_redirects=False)

    def downloader():  # 按读到的版本号下载:内容必须与该版本标签严格对应
        c = TestClient(app)
        while not stop.is_set():
            rev = c.get(f"/api/runs/{rid}").json()["label_revision"]
            resp = c.get(f"/runs/{rid}/download/v{rev}/S1_R1.fastq")
            if resp.status_code == 200:
                # 偶数版:S1=原 S1;奇数版:S1=原 S2。错一点就是标签混杂
                expect = orig["S2"] if rev % 2 else orig["S1"]
                if resp.text != expect:
                    errors.append(f"v{rev}: 内容与标签版本不符")
                    stop.set()
            elif resp.status_code != 410:  # 410 = 版本刚被换掉,属正常
                errors.append(f"v{rev}: 意外状态 {resp.status_code}")
                stop.set()

    threads = [threading.Thread(target=swapper), threading.Thread(target=downloader)]
    for t in threads:
        t.start()
    time.sleep(2.0)
    stop.set()
    for t in threads:
        t.join(timeout=30)
    assert not errors

    # 交错结束后状态仍自洽:当前版本可下,上一版已失效
    st = client.get(f"/api/runs/{rid}").json()
    rev = st["label_revision"]
    assert rev >= 1
    resp = client.get(f"/runs/{rid}/download/v{rev}/S1_R1.fastq")
    assert resp.status_code == 200
    assert client.get(
        f"/runs/{rid}/download/v{rev - 1}/S1_R1.fastq").status_code == 410


def test_uncorrected_run_keeps_plain_download_urls(client):
    """未更正运行保持原有下载方式不变。"""
    rid = published_run(client, "RUNC9")
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 200
    assert client.get(f"/runs/{rid}/download/v0/S1_R1.fastq").status_code == 200
    assert client.get(f"/runs/{rid}/download/v1/S1_R1.fastq").status_code == 404
    assert client.get(f"/runs/{rid}/download/demux.log").status_code == 200
    page = client.get(f"/runs/{rid}").text
    assert f'href="/runs/{rid}/download/S1_R1.fastq"' in page
    assert "样本号更正" in page  # 更正表单在运行页可用

