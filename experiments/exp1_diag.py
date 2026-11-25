import glob, os, re, json
import numpy as np, scipy.io as sio
from scipy import signal
from dsp import *

ROOT = os.environ.get("CWRU_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CWRU-dataset"))
OUT = os.path.dirname(os.path.abspath(__file__)) + "/results"
os.makedirs(OUT, exist_ok=True)
WIN = 12000  # 1 s windows


def load(path):
    m = sio.loadmat(path)
    de = [k for k in m if k.endswith("_DE_time")][0]
    x = m[de].ravel().astype(float)
    rpmk = [k for k in m if k.endswith("RPM")]
    return x, (float(m[rpmk[0]].ravel()[0]) if rpmk else None)


def records():
    recs = []
    for p in sorted(glob.glob(ROOT + "/Normal/*.mat")):
        load_hp = int(re.search(r"_(\d)\.mat", p).group(1))
        recs.append(dict(path=p, cls="Normal", size="-", pos="-", load=load_hp, fs_native=48000))
    for p in sorted(glob.glob(ROOT + "/12k_Drive_End_Bearing_Fault_Data/*/*/**/*.mat", recursive=True)):
        parts = p.split("/")
        i = parts.index("12k_Drive_End_Bearing_Fault_Data")
        cls, size = parts[i + 1], parts[i + 2]
        if size == "028":
            continue  # 0.028" faults use a different (NTN) bearing, excluded per Smith & Randall
        pos = "@6"
        m = re.search(r"@(\d+)", p)
        if cls == "OR" and m:
            pos = "@" + m.group(1)
        load_hp = int(re.search(r"_(\d)\.mat$", p).group(1))
        cls = {"B": "Ball"}.get(cls, cls)
        recs.append(dict(path=p, cls=cls, size=size, pos=pos if cls == "OR" else "-", load=load_hp, fs_native=12000))
    return recs


def windows(rec):
    x, rpm = load(rec["path"])
    if rec["fs_native"] == 48000:
        x = signal.decimate(x, 4, ftype="fir", zero_phase=True)
    rpm = rpm if rpm and rpm > 1000 else NOMINAL_RPM[rec["load"]]
    nw = len(x) // WIN
    return [x[k * WIN:(k + 1) * WIN] for k in range(nw)], rpm


if __name__ == "__main__":
    rows = []
    for r in records():
        ws, rpm = windows(r)
        for k, w in enumerate(ws):
            pred, sc, tf, band = classify(w, rpm)
            rows.append(dict(file=os.path.basename(r["path"]), cls=r["cls"], size=r["size"], pos=r["pos"],
                             load=r["load"], rpm=rpm, win=k, pred=pred, **{f"snr_{a}": b for a, b in sc.items()},
                             **tf, vel_mms=velocity_rms_mms(w), band_lo=band[0], band_hi=band[1], fc=band[2]))
    json.dump(rows, open(OUT + "/diag_rows.json", "w"))
    import collections
    print("files", len(set(r["file"] for r in rows)), "windows", len(rows))
    cm = collections.Counter((r["cls"], r["pred"]) for r in rows)
    labs = ["Normal", "IR", "OR", "Ball"]
    print("true\\pred", labs)
    for t in labs:
        print(t, [cm[(t, p)] for p in labs])
    acc = sum(r["cls"] == r["pred"] for r in rows) / len(rows)
    print("acc", acc)
