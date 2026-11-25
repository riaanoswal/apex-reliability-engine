"""Full diagnostic evaluation: APEX rule engine vs baselines on real CWRU data."""
import json, collections, time
import numpy as np
from scipy import signal
from dsp import *
from exp1_diag import records, windows, OUT
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

LABS = ["Normal", "IR", "OR", "Ball"]
rng = np.random.default_rng(0)

# ---- feature extraction for every window, several pipelines ----
data = []
for r in records():
    ws, rpm = windows(r)
    fr = rpm / 60
    for k, w in enumerate(ws):
        tf = time_features(w)
        # A: SK-selected band envelope (APEX)
        f, E, band = envelope_spectrum(w)
        scA = fault_scores(f, E, fr)
        # B: fixed band envelope 2-4 kHz (common default; the prototype's "2-4 kHz" wording)
        f2, E2, _ = envelope_spectrum(w, band=(2000, 4000))
        scB = fault_scores(f2, E2, fr)
        # C: raw (non-demodulated) amplitude spectrum
        n = len(w)
        R = np.abs(np.fft.rfft((w - w.mean()) * np.hanning(n))) / n
        fR = np.fft.rfftfreq(n, 1 / FS)
        scC = fault_scores(fR, R, fr)
        data.append(dict(file=r["path"].split("/")[-1], cls=r["cls"], size=r["size"], pos=r["pos"], load=r["load"],
                         rpm=rpm, tf=tf, A=scA, B=scB, C=scC, vel=velocity_rms_mms(w), band=band))

print("windows", len(data))


def rule(sc, thr):
    b = max(sc, key=sc.get)
    return "Normal" if sc[b] < thr else {"BPFO": "OR", "BPFI": "IR", "BSF": "Ball"}[b]


def evaluate(pred, true):
    return dict(acc=float(np.mean(np.array(pred) == np.array(true))),
                bacc=float(balanced_accuracy_score(true, pred)),
                f1=float(f1_score(true, pred, labels=LABS, average="macro", zero_division=0)),
                cm=confusion_matrix(true, pred, labels=LABS).tolist())


def detect_eval(pred_fault, true):
    tf = np.array([t != "Normal" for t in true]); pf = np.array(pred_fault)
    tp = np.sum(tf & pf); fp = np.sum(~tf & pf); fn = np.sum(tf & ~pf); tn = np.sum(~tf & ~pf)
    return dict(tpr=float(tp / (tp + fn)), fpr=float(fp / (fp + tn)), prec=float(tp / max(tp + fp, 1)),
                acc=float((tp + tn) / len(tf)))


results = {}
train = [d for d in data if d["load"] == 0]
test = [d for d in data if d["load"] != 0]
ytest = [d["cls"] for d in test]

# calibrate threshold on load-0 only
thr_grid = np.arange(4, 16.01, 0.5)
for key in ("A", "B", "C"):
    best = max(thr_grid, key=lambda t: balanced_accuracy_score([d["cls"] for d in train], [rule(d[key], t) for d in train]))
    pred = [rule(d[key], best) for d in test]
    results[key] = dict(thr=float(best), **evaluate(pred, ytest),
                        det=detect_eval([p != "Normal" for p in pred], ytest))
    # also 3-class (exclude Ball) as Smith & Randall treat ball faults separately
    sub = [(d, p) for d, p in zip(test, pred) if d["cls"] != "Ball"]
    results[key]["acc_noball"] = float(np.mean([d["cls"] == p for d, p in sub]))
    print(key, "thr", best, {k: round(v, 3) for k, v in results[key].items() if isinstance(v, float)}, results[key]["det"])

# kurtosis-only detector (classic kappa>3 health indicator; and calibrated)
for thr in (3.0, None):
    if thr is None:
        grid = np.arange(2.5, 8.01, 0.1)
        thr = float(max(grid, key=lambda t: balanced_accuracy_score([d["cls"] != "Normal" for d in train], [d["tf"]["kurtosis"] > t for d in train])))
    res = detect_eval([d["tf"]["kurtosis"] > thr for d in test], ytest)
    results[f"kurt_{thr:.1f}"] = res
    print("kurtosis>", thr, res)

# data-driven baseline: random forest, leave-one-load-out
def feats(d, use_env):
    t = d["tf"]
    v = [t["rms_g"], t["peak_g"], t["crest"], t["kurtosis"], t["skew"]]
    if use_env:
        v += [d["A"]["BPFO"], d["A"]["BPFI"], d["A"]["BSF"]]
    return v

for use_env in (False, True):
    preds, trues = [], []
    for L in (1, 2, 3):  # train on load 0 + other loads except L? -> leave-one-load-out among all 4
        tr = [d for d in data if d["load"] != L]; te = [d for d in data if d["load"] == L]
        clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
        clf.fit([feats(d, use_env) for d in tr], [d["cls"] for d in tr])
        preds += list(clf.predict([feats(d, use_env) for d in te])); trues += [d["cls"] for d in te]
    k = "RF_env" if use_env else "RF_time"
    results[k] = evaluate(preds, trues)
    print(k, {a: round(b, 3) for a, b in results[k].items() if isinstance(b, float)})

# leave-one-FILE-group-out (fault size) for RF, to test generalisation to unseen defect sizes
for use_env in (False, True):
    preds, trues = [], []
    for S in ("007", "014", "021"):
        tr = [d for d in data if d["size"] != S]; te = [d for d in data if d["size"] == S]
        clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
        clf.fit([feats(d, use_env) for d in tr], [d["cls"] for d in tr])
        preds += list(clf.predict([feats(d, use_env) for d in te])); trues += [d["cls"] for d in te]
    k = ("RF_env" if use_env else "RF_time") + "_unseen_size"
    results[k] = dict(acc=float(np.mean(np.array(preds) == np.array(trues))), n=len(trues))
    print(k, results[k])
# APEX rule on the same unseen-size fault windows (rule has no training, use thr A)
fa = [d for d in data if d["size"] in ("007", "014", "021")]
results["A_faultsonly_acc"] = float(np.mean([rule(d["A"], results["A"]["thr"]) == d["cls"] for d in fa]))

# per-condition table for APEX (all loads, calibrated thr)
thrA = results["A"]["thr"]
g = collections.defaultdict(list)
for d in data:
    g[(d["cls"], d["size"], d["pos"])].append(d)
per = []
for k, v in sorted(g.items()):
    p = [rule(d["A"], thrA) for d in v]
    per.append(dict(cls=k[0], size=k[1], pos=k[2], n=len(v), acc=float(np.mean([x == k[0] for x in p])),
                    kurt=float(np.mean([d["tf"]["kurtosis"] for d in v])), crest=float(np.mean([d["tf"]["crest"] for d in v])),
                    vel=float(np.mean([d["vel"] for d in v])), rms=float(np.mean([d["tf"]["rms_g"] for d in v])),
                    snrO=float(np.mean([d["A"]["BPFO"] for d in v])), snrI=float(np.mean([d["A"]["BPFI"] for d in v])),
                    snrB=float(np.mean([d["A"]["BSF"] for d in v]))))
results["per_condition"] = per
for p in per:
    print(p)

# full-data confusion for APEX (all loads) for figure
allp = [rule(d["A"], thrA) for d in data]
results["A_all"] = evaluate(allp, [d["cls"] for d in data])
# per-load accuracy
results["per_load"] = {L: float(np.mean([rule(d["A"], thrA) == d["cls"] for d in data if d["load"] == L])) for L in range(4)}
results["n_windows"] = len(data); results["n_files"] = len(set(d["file"] for d in data))
results["class_counts"] = dict(collections.Counter(d["cls"] for d in data))
json.dump(results, open(OUT + "/eval.json", "w"), indent=1)
print(results["per_load"], results["class_counts"])
