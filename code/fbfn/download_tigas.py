"""从 HuggingFace 下载 TIGAS 数据集 sd14（SD v1.4）子集的一部分到项目内。

目录结构：train/images/sd14/{0_real,1_fake}/*.{jpg,jpeg,png}
下载后按 train/val 分流到：
    <研究项目>/data/tigas/{train,val}/{real,fake}/

默认：真实/生成各取前 3500 张（train 3000 + val 500）。
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

REPO = "H1merka/TIGAS_dataset"
API_BASE = "https://hf-mirror.com/api/datasets/" + REPO + "/tree/main/"
RES_BASE = "https://hf-mirror.com/datasets/" + REPO + "/resolve/main/"
# TIGAS 数据根目录：<研究项目>/data/tigas（随项目一起移动，不再硬编码盘符）
DST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "tigas",
)
N_TRAIN = 3000
N_VAL = 500
N = N_TRAIN + N_VAL
WORKERS = 16


def list_files(sub):
    """用 HTTP tree API + cursor 翻页拿全列表，返回排序后的文件路径。"""
    files = []
    url = API_BASE + sub + "?recursive=true&expand=false&limit=1000"
    while url:
        # 把 Link header 里的 huggingface.co 换成 hf-mirror.com
        url = url.replace("huggingface.co", "hf-mirror.com")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        for it in r.json():
            if it.get("type") == "file":
                files.append(it["path"])
        url = None
        link = r.headers.get("Link", "")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        if m:
            url = m.group(1)
    return sorted(files)


def download_one(path, dst_dir):
    dst = os.path.join(dst_dir, os.path.basename(path))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    for _ in range(3):
        try:
            r = requests.get(RES_BASE + path, timeout=180)  # 自动跟随 LFS 重定向
            r.raise_for_status()
            with open(dst, "wb") as f:
                f.write(r.content)
            return "ok"
        except Exception:
            continue
    return "fail"


def main():
    real_files = list_files("train/images/sd14/0_real")
    fake_files = list_files("train/images/sd14/1_fake")
    print(f"sd14 真实图 {len(real_files)} 张，生成图 {len(fake_files)} 张", flush=True)

    real_pick = real_files[:N]
    fake_pick = fake_files[:N]
    print(f"本次下载：真实 {len(real_pick)}、生成 {len(fake_pick)}（各 train {N_TRAIN} + val {N_VAL}）",
          flush=True)

    tasks = []  # (remote_path, dst_dir)
    for i, p in enumerate(real_pick):
        sub = "train/real" if i < N_TRAIN else "val/real"
        tasks.append((p, os.path.join(DST, sub)))
    for i, p in enumerate(fake_pick):
        sub = "train/fake" if i < N_TRAIN else "val/fake"
        tasks.append((p, os.path.join(DST, sub)))

    for d in (os.path.join(DST, "train/real"), os.path.join(DST, "train/fake"),
              os.path.join(DST, "val/real"), os.path.join(DST, "val/fake")):
        os.makedirs(d, exist_ok=True)

    ok = skip = fail = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(download_one, p, d): p for p, d in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            if r == "ok":
                ok += 1
            elif r == "skip":
                skip += 1
            else:
                fail += 1
            if (ok + skip + fail) % 200 == 0:
                print(f"  进度 {ok + skip + fail}/{len(tasks)} (ok {ok} skip {skip} fail {fail})",
                      flush=True)

    print(f"完成：ok {ok}，skip {skip}，fail {fail}", flush=True)


if __name__ == "__main__":
    main()
