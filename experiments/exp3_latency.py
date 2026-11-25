"""Measured CPU latency of each APEX pipeline stage + flagship case (CWRU 235, OR021@6, 1 HP, 1772 rpm)."""
import json, time, platform
import numpy as np, scipy, sklearn
from dsp import *
from exp1_diag import load, OUT, ROOT
from retrieval import build_corpus, TfidfRetriever

x, rpm = load(ROOT + "/12k_Drive_End_Bearing_Fault_Data/OR/021/@6/235_1.mat")
rpm = rpm or 1772.0
fr = rpm / 60
w = x[:12000]

# ---- flagship metrics over all 1-s windows of file 235 ----
flag = []
for k in range(len(x) // 12000):
    s = x[k * 12000:(k + 1) * 12000]
    f, E, band = envelope_spectrum(s)
    snr, fpk = peak_snr_db(f, E, MULT["BPFO"] * fr)
    tf = time_features(s)
    flag.append(dict(kurt=tf["kurtosis"], crest=tf["crest"], rms=tf["rms_g"], peak=tf["peak_g"],
                     vel=velocity_rms_mms(s), bpfo_meas=fpk, bpfo_snr=snr, fc=band[2]))
agg = {k: (float(np.mean([d[k] for d in flag])), float(np.std([d[k] for d in flag], ddof=1))) for k in flag[0]}
agg["n_windows"] = len(flag); agg["rpm"] = rpm; agg["fr"] = fr; agg["bpfo_theory"] = MULT["BPFO"] * fr
print(json.dumps(agg, indent=1))

# ---- stage latencies ----
docs = build_corpus()
ret = TfidfRetriever(docs, ngram=(1, 2))


def timeit(fn, n=300):
    for _ in range(10):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return dict(median=float(np.median(ts)), p95=float(np.percentile(ts, 95)), mean=float(ts.mean()))

stages = {
    "time_features": lambda: time_features(w),
    "velocity_rms": lambda: velocity_rms_mms(w),
    "welch_psd": lambda: welch_psd(w),
    "sk_band_select": lambda: select_band(w),
    "envelope_spectrum": lambda: envelope_spectrum(w),
    "rule_classify_total": lambda: classify(w, rpm),
    "tfidf_retrieval": lambda: ret.query("outer race bearing spall BPFO high kurtosis drive end", k=3),
}
from fallback import deterministic_report
stages["deterministic_report"] = lambda: deterministic_report("OR", 21.8, 105.9, 0.40, docs[0])
lat = {k: timeit(v) for k, v in stages.items()}
for k, v in lat.items():
    print(f"{k:24s} median {v['median']:.3f} ms  p95 {v['p95']:.3f} ms")
env = dict(cpu=platform.processor() or "Intel Xeon @ 2.80GHz (2 vCPU)", python=platform.python_version(),
           numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, corpus=len(docs))
json.dump(dict(flagship=agg, latency=lat, env=env), open(OUT + "/latency.json", "w"), indent=1)
