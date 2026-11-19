"""CPU only: extract 16 kHz mono wav from mp4 for datasets that ship video only
(MUSIC-AVQA, AVQA, AVHBench). CMM already ships wav.

  python prepare_audio.py --dataset music_avqa --workers 16
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import data as D

SR = 16000


def extract(job):
    src, dst = job
    if os.path.exists(dst) or not os.path.exists(src):
        return dst if os.path.exists(dst) else None
    import av
    import numpy as np
    import soundfile as sf
    try:
        with av.open(src) as c:
            if not c.streams.audio:
                return None
            rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
            chunks = []
            for fr in c.decode(audio=0):
                for o in rs.resample(fr):
                    chunks.append(o.to_ndarray().reshape(-1))
            for o in rs.resample(None):
                chunks.append(o.to_ndarray().reshape(-1))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sf.write(dst + ".tmp.wav", np.concatenate(chunks) if chunks else np.zeros(SR, "float32"), SR)
        os.replace(dst + ".tmp.wav", dst)
        return dst
    except Exception as e:
        print("FAIL", src, e, flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["music_avqa", "music_val", "avqa_mcq", "avqa_dev", "avhbench", "avh_cap"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    recs = D.load(a.dataset)
    jobs = sorted({(r["video"], r["audio"]) for r in recs})
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(extract, jobs, chunksize=16))
    ok = sum(r is not None for r in res)
    missing = [j[0] for j, r in zip(jobs, res) if r is None]
    print(f"{a.dataset}: {ok}/{len(jobs)} wav ready, {len(missing)} missing/failed")
    if missing:
        with open(f"{D.AUDIO_CACHE}/{a.dataset}_missing.txt", "w") as f:
            f.write("\n".join(missing))


if __name__ == "__main__":
    main()
