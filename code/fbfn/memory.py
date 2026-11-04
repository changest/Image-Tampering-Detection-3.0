# -*- coding: utf-8 -*-
"""结构化案例记忆（Jev-Mem 借鉴的 MVP）。

- SQLite 持久化：cases(feature, label, type, source, meta)
- 廉价特征：ELA 统计 + 熵图统计 + 颜色矩（约 30 维，无需模型）
- 检索：余弦相似度暴力 KNN（几万条以内足够快）
- 用途：新图先查相似历史案例，用邻居标签校准系统一置信度，决定是否升级全模型

用法示例：
    mem = CaseMemory("data/memory/cases.sqlite")
    mem.add_many(feats, labels, types, sources)
    sims, idx = mem.query(feat, k=10)
"""
import os
import sqlite3
from collections import Counter

import numpy as np


def extract_feature(x, ent, ela):
    """x: (3,H,W) [0,1]；ent: (3,31,31)；ela: (1,H,W)。返回 L2 归一化特征向量。"""
    x = np.asarray(x, dtype=np.float32)
    ent = np.asarray(ent, dtype=np.float32).reshape(ent.shape[0], -1)
    ela = np.asarray(ela, dtype=np.float32).reshape(-1)

    # ELA 统计：分位数 + 均值/标准差 + 8 档直方图
    ela_q = np.quantile(ela, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    ela_m = np.array([ela.mean(), ela.std()])
    hist, _ = np.histogram(ela, bins=8, range=(0.0, max(0.05, float(ela.max()))))
    hist = hist.astype(np.float32) / max(1.0, ela.size)
    ela_f = np.concatenate([ela_q, ela_m, hist])

    # 熵图统计：每通道均值/标准差
    ent_f = np.concatenate([ent.mean(axis=1), ent.std(axis=1)])

    # 颜色矩：每通道均值/标准差/三阶矩
    mean = x.mean(axis=(1, 2))
    std = x.std(axis=(1, 2))
    skew = (((x - mean[:, None, None]) / (std[:, None, None] + 1e-6)) ** 3).mean(axis=(1, 2))
    color_f = np.concatenate([mean, std, skew])

    v = np.concatenate([ela_f, ent_f, color_f]).astype(np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n


class CaseMemory:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cases ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "feature BLOB NOT NULL,"
            "label INTEGER NOT NULL,"
            "type INTEGER NOT NULL DEFAULT -1,"
            "source TEXT DEFAULT '',"
            "meta TEXT DEFAULT '')")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS relations ("
            "src INTEGER NOT NULL, dst INTEGER NOT NULL, sim REAL NOT NULL)")
        self.conn.commit()
        self._cache = None

    def add_many(self, feats, labels, types=None, sources=None):
        feats = np.asarray(feats, dtype=np.float32)
        n = len(feats)
        types = types if types is not None else [-1] * n
        sources = sources if sources is not None else [""] * n
        rows = [(f.tobytes(), int(l), int(t), str(s), "")
                for f, l, t, s in zip(feats, labels, types, sources)]
        self.conn.executemany(
            "INSERT INTO cases(feature, label, type, source, meta) VALUES (?,?,?,?,?)", rows)
        self.conn.commit()
        self._cache = None

    def _load(self):
        if self._cache is not None:
            return self._cache
        cur = self.conn.execute("SELECT id, feature, label, type, source FROM cases")
        ids, feats, labels, types, srcs = [], [], [], [], []
        for cid, blob, lab, typ, src in cur:
            v = np.frombuffer(blob, dtype=np.float32)
            if v.size == 0:
                continue
            ids.append(cid); feats.append(v); labels.append(lab)
            types.append(typ); srcs.append(src)
        if feats:
            F = np.stack(feats)
            F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        else:
            F = np.zeros((0, 1), np.float32)
        self._cache = (np.array(ids), F, np.array(labels), np.array(types), srcs)
        return self._cache

    def __len__(self):
        return len(self._load()[0])

    def query(self, feat, k=10):
        """返回 (sims, labels, types) 各长度 k。"""
        ids, F, labels, types, _ = self._load()
        if len(ids) == 0:
            return np.zeros(0, np.float32), np.zeros(0, int), np.zeros(0, int)
        f = np.asarray(feat, dtype=np.float32)
        f = f / (np.linalg.norm(f) + 1e-8)
        sims = F @ f
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        order = idx[np.argsort(-sims[idx])]
        return sims[order], labels[order], types[order]

    def stats(self):
        _, _, labels, types, _ = self._load()
        return {"n": int(len(labels)),
                "labels": dict(Counter(labels.tolist())),
                "types": dict(Counter(types.tolist()))}

    def close(self):
        self.conn.close()
