import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import signal
from dsp import *
from exp1_diag import load, ROOT

FIG = "figs"; os.makedirs(FIG, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
                     "savefig.dpi": 220, "savefig.bbox": "tight", "legend.frameon": False})
E = json.load(open("results/eval.json")); L = json.load(open("results/latency.json"))
F = json.load(open("results/fallback.json")); Rt = json.load(open("results/retrieval.json"))
P = lambda s: ROOT + "/12k_Drive_End_Bearing_Fault_Data/" + s
files = {"Normal (97, 0 HP)": (ROOT + "/Normal/98_Normal_1.mat", True), "Inner race 0.021\" (210, 1 HP)": (P("IR/021/210_1.mat"), False),
         "Outer race 0.021\" @6 (235, 1 HP)": (P("OR/021/@6/235_1.mat"), False), "Ball 0.021\" (223, 1 HP)": (P("B/021/223_1.mat"), False)}
files["Normal (98, 1 HP)"] = files.pop("Normal (97, 0 HP)")
sig = {}
for k, (p, dec) in files.items():
    x, rpm = load(p)
    if dec: x = signal.decimate(x, 4, ftype="fir", zero_phase=True)
    sig[k] = (x[:12000], rpm or 1772.0)
order = ["Normal (98, 1 HP)", "Inner race 0.021\" (210, 1 HP)", "Outer race 0.021\" @6 (235, 1 HP)", "Ball 0.021\" (223, 1 HP)"]

# Fig: architecture
fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 62)
def box(x, y, w, h, t, s="", fc="#f3f6fb", ec=C[0]):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2 + (1.6 if s else 0), t, ha="center", va="center", fontsize=8.5, weight="bold", color=INK)
    if s: ax.text(x + w / 2, y + h / 2 - 2.2, s, ha="center", va="center", fontsize=6.8, color=INK2)
def arr(x1, y1, x2, y2, t=""):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, t, fontsize=6.5, color=INK2, ha="center")
box(1, 46, 17, 12, "L1 Acquisition", "12 kHz DE accel.\nCWRU / SCADA", ec=C[2], fc="#eefaf5")
box(23, 46, 20, 12, "L2 Kinematic DSP", "kurtosis, SK band,\nenvelope, BPFO/BPFI/BSF")
box(48, 46, 20, 12, "L3 CMMS Retrieval", "TF-IDF / BM25 over\nISO 14224 records")
box(73, 46, 26, 12, "L4 Hybrid Dispatcher", "circuit breaker +\nNeMo-style output checks")
box(60, 24, 18, 12, "Path A: Cloud LLM", "Gemini API (RAG\nprompt, T=0.1)", ec=C[1], fc="#fdf1ec")
box(81, 24, 18, 12, "Path B: Edge", "deterministic rules\n/ local LLM (Ollama)", ec=C[1], fc="#fdf1ec")
box(23, 24, 32, 12, "Risk & Work-Order Synthesis", "RPN = S x O x D, ISO 10816 zone,\nSAP PM BAPI JSON", ec=C[6], fc="#f2f0fb")
box(1, 2, 98, 13, "L5 Presentation: FastAPI REST  ->  React/TypeScript SCADA dashboard (ISA-95 Level 2/3 HMI)", "KPI strip | waveform & PSD | reliability terminal | kernel trace | work-order export", ec=C[3], fc="#fdf7e8")
arr(18, 52, 23, 52); arr(43, 52, 48, 52); arr(68, 52, 73, 52)
arr(82, 46, 70, 36); arr(88, 46, 90, 36); arr(60, 30, 55, 30); arr(81, 27, 78, 27)
arr(33, 46, 36, 36, "class, kappa"); arr(39, 24, 39, 15)
ax.text(90, 40.5, "on 429/404/\ntimeout", fontsize=6.3, color=C[7], ha="left")
fig.savefig(FIG + "/arch.png"); plt.close()

# Fig: defect frequencies vs speed
rpm = np.linspace(600, 3600, 50)
fig, ax = plt.subplots(figsize=(6.5, 2.8))
for i, (n, m) in enumerate([("BPFI", 5.4152), ("2xBSF", 2 * 4.7135), ("BPFO", 3.5848), ("1x shaft", 1.0), ("FTF", 0.39828)]):
    ax.plot(rpm, m * rpm / 60, color=C[i], lw=1.6); ax.text(3620, m * 3600 / 60, n, va="center", fontsize=8, color=INK)
ax.axvline(1772, color=INK2, ls="--", lw=0.8); ax.text(1790, 540, "1772 rpm\n(CWRU 1 HP)", fontsize=7.5, color=INK2)
ax.set_xlabel("Shaft speed (rpm)"); ax.set_ylabel("Frequency (Hz)"); ax.set_xlim(600, 3900)
fig.savefig(FIG + "/defect_freqs.png"); plt.close()

# Fig: waveforms
fig, axs = plt.subplots(4, 1, figsize=(6.8, 5.6), sharex=True)
t = np.arange(12000) / FS * 1000
for i, k in enumerate(order):
    x = sig[k][0]; axs[i].plot(t[:1800], x[:1800], color=C[0], lw=0.6)
    tf = time_features(x)
    axs[i].set_title(f"{k}   kurtosis = {tf['kurtosis']:.2f}, crest = {tf['crest']:.2f}", fontsize=8.5, loc="left", color=INK)
    axs[i].set_ylabel("g")
axs[-1].set_xlabel("Time (ms)"); fig.tight_layout(); fig.savefig(FIG + "/waveforms.png"); plt.close()

# Fig: PSD vs envelope for OR
x, r = sig[order[2]]; fr = r / 60; bp = MULT["BPFO"] * fr
fig, axs = plt.subplots(2, 1, figsize=(6.8, 4.6))
f, Pxx = welch_psd(x)
axs[0].semilogy(f, Pxx, color=C[0], lw=0.7); axs[0].set_xlim(0, 6000); axs[0].set_ylabel("PSD (g$^2$/Hz)")
axs[0].set_title("(a) Welch PSD (Hann, 4096 pts, 50% overlap): energy concentrates in 2-5 kHz structural resonances", fontsize=8.5, loc="left")
fe, Ee, band = envelope_spectrum(x)
axs[0].axvspan(band[0], band[1], color=C[3], alpha=0.18, lw=0); axs[0].text(band[1] + 60, Pxx.max() / 3, "SK-selected\ndemodulation band", fontsize=7.5)
axs[1].plot(fe, Ee * 1000, color=C[0], lw=0.8); axs[1].set_xlim(0, 500)
for h in range(1, 5):
    axs[1].axvline(h * bp, color=C[7], ls=":", lw=0.9)
axs[1].axvline(fr, color=C[2], ls=":", lw=0.9)
axs[1].text(bp + 3, axs[1].get_ylim()[1] * 0.9, f"BPFO = {bp:.1f} Hz (theory)", fontsize=7.5, color=INK)
axs[1].text(fr + 3, axs[1].get_ylim()[1] * 0.6, "1x", fontsize=7.5, color=INK)
axs[1].set_xlabel("Frequency (Hz)"); axs[1].set_ylabel("Envelope amplitude (mg)")
axs[1].set_title(f"(b) Envelope spectrum: BPFO and harmonics (dotted) at 1x-4x", fontsize=8.5, loc="left")
fig.tight_layout(); fig.savefig(FIG + "/psd_env_or.png"); plt.close()

# Fig: IR envelope with sidebands
x, r = sig[order[1]]; fr = r / 60; bi = MULT["BPFI"] * fr
fe, Ee, band = envelope_spectrum(x)
fig, ax = plt.subplots(figsize=(6.8, 2.6))
ax.plot(fe, Ee * 1000, color=C[0], lw=0.8); ax.set_xlim(0, 500)
for h in (1, 2):
    ax.axvline(h * bi, color=C[7], ls=":", lw=0.9)
    for s in (-1, 1): ax.axvline(h * bi + s * fr, color=C[1], ls="--", lw=0.7)
ax.text(bi + 3, ax.get_ylim()[1] * 0.9, f"BPFI = {bi:.1f} Hz", fontsize=7.5)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Envelope (mg)")
fig.savefig(FIG + "/env_ir.png"); plt.close()

# Fig: spectral kurtosis
fig, ax = plt.subplots(figsize=(6.8, 2.6))
for i, k in enumerate(order):
    f, sk = stft_spectral_kurtosis(sig[k][0]); ax.plot(f, sk, color=C[i], lw=1.1, label=k)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Spectral kurtosis"); ax.set_xlim(0, 5500); ax.set_ylim(-1, 8); ax.legend(fontsize=7, ncol=2)
fig.savefig(FIG + "/sk.png"); plt.close()

# Fig: confusion matrices
labs = ["Normal", "IR", "OR", "Ball"]
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0))
for ax, (cm, title) in zip(axs, [(E["A_all"]["cm"], "(a) APEX rule engine, all loads (n=555)"), (E["A"]["cm"], "(b) APEX, held-out loads 1-3 (n=420)")]):
    cm = np.array(cm); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j] * 100:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if cmn[i, j] > .6 else INK)
    ax.set_xticks(range(4), labs); ax.set_yticks(range(4), labs); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG + "/cm.png"); plt.close()

# Fig: per-condition accuracy
pc = E["per_condition"]
lab = [f"{p['cls']} {p['size']} {p['pos'] if p['pos'] != '-' else ''}".replace(" -", "").strip() for p in pc]
fig, ax = plt.subplots(figsize=(6.8, 2.8))
cols = [C[7] if p["acc"] < .5 else C[0] for p in pc]
ax.bar(range(len(pc)), [p["acc"] * 100 for p in pc], color=cols, width=0.7)
for i, p in enumerate(pc): ax.text(i, p["acc"] * 100 + 2, f"{p['acc'] * 100:.0f}", ha="center", fontsize=7)
ax.set_xticks(range(len(pc)), lab, rotation=40, ha="right", fontsize=7.5); ax.set_ylabel("Window accuracy (%)"); ax.set_ylim(0, 112)
fig.savefig(FIG + "/per_condition.png"); plt.close()

# Fig: kurtosis by class boxplot
rows = json.load(open("results/diag_rows.json"))
fig, ax = plt.subplots(figsize=(6.8, 2.6))
data = [[r["kurtosis"] for r in rows if r["cls"] == c] for c in labs]
bp_ = ax.boxplot(data, tick_labels=labs, widths=0.5, patch_artist=True, medianprops=dict(color=INK))
for b, c in zip(bp_["boxes"], C): b.set_facecolor(c); b.set_alpha(0.55)
ax.axhline(3, color=INK2, ls="--", lw=0.8); ax.text(4.35, 3.4, "Gaussian (3)", fontsize=7)
ax.set_yscale("log"); ax.set_ylabel("Kurtosis (log)")
fig.savefig(FIG + "/kurt_box.png"); plt.close()

# Fig: latency
lat = L["latency"]; names = ["time_features", "velocity_rms", "welch_psd", "sk_band_select", "envelope_spectrum", "rule_classify_total", "tfidf_retrieval"]
nice = ["Time features", "Velocity RMS (ISO band)", "Welch PSD", "SK band selection", "Envelope spectrum", "Rule classifier (total)", "TF-IDF retrieval (48 docs)"]
fig, ax = plt.subplots(figsize=(6.8, 2.6))
y = np.arange(len(names))
ax.barh(y, [lat[n]["median"] for n in names], color=C[0], height=0.55, label="median")
ax.scatter([lat[n]["p95"] for n in names], y, color=C[1], s=14, zorder=3, label="p95")
for i, n in enumerate(names): ax.text(lat[n]["p95"] + 0.15, i, f"{lat[n]['median']:.2f}", va="center", fontsize=7)
ax.set_yticks(y, nice); ax.invert_yaxis(); ax.set_xlabel("Latency per 1-s window (ms), CPU"); ax.legend(fontsize=7, loc="lower right")
fig.savefig(FIG + "/latency.png"); plt.close()

# Fig: fallback
sc = []
for r in F:
    if r["scenario"] not in sc: sc.append(r["scenario"])
fig, ax = plt.subplots(figsize=(6.8, 2.8)); w = 0.38; x_ = np.arange(len(sc))
for j, (s, c) in enumerate([("naive", C[1]), ("breaker", C[0])]):
    v = [next(r for r in F if r["scenario"] == k and r["strategy"] == s)["p95_ms"] for k in sc]
    ax.bar(x_ + (j - .5) * w, v, w, color=c, label={"naive": "Prototype dispatcher", "breaker": "Circuit-breaker dispatcher"}[s])
ax.set_yscale("log"); ax.set_ylabel("p95 time to report (ms)"); ax.set_xticks(x_, sc, rotation=25, ha="right", fontsize=7.5); ax.legend(fontsize=7)
fig.savefig(FIG + "/fallback.png"); plt.close()

# Fig: retrieval
fig, ax = plt.subplots(figsize=(6.8, 2.6)); meths = ["tfidf_uni", "tfidf_unibi", "bm25"]; augs = ["none", "static", "dsp"]
for j, a in enumerate(augs):
    ax.bar(np.arange(3) + (j - 1) * 0.26, [Rt[f"{m}|{a}"]["top1"] * 100 for m in meths], 0.26, color=C[j],
           label={"none": "Query only", "static": "Static prototype suffix", "dsp": "DSP-conditioned suffix"}[a])
ax.set_xticks(range(3), ["TF-IDF (1-gram)", "TF-IDF (1,2-gram)", "BM25"]); ax.set_ylabel("Top-1 accuracy (%)"); ax.set_ylim(0, 105); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16)); ax.set_ylim(0, 105)
fig.savefig(FIG + "/retrieval.png"); plt.close()

# Fig: threshold sensitivity (all windows)
from sklearn.metrics import balanced_accuracy_score
def rule(sc_, thr):
    b = max(sc_, key=sc_.get); return "Normal" if sc_[b] < thr else {"BPFO": "OR", "BPFI": "IR", "BSF": "Ball"}[b]
thr = np.arange(3, 20.1, 0.5); true = [r["cls"] for r in rows]
ba, fpr, tpr = [], [], []
for th in thr:
    pr = [rule({"BPFO": r["snr_BPFO"], "BPFI": r["snr_BPFI"], "BSF": r["snr_BSF"]}, th) for r in rows]
    ba.append(balanced_accuracy_score(true, pr) * 100)
    fpr.append(np.mean([p != "Normal" for p, t in zip(pr, true) if t == "Normal"]) * 100)
    tpr.append(np.mean([p != "Normal" for p, t in zip(pr, true) if t != "Normal"]) * 100)
fig, ax = plt.subplots(figsize=(6.8, 2.6))
ax.plot(thr, ba, color=C[0], lw=1.6, label="Balanced accuracy (4-class)")
ax.plot(thr, tpr, color=C[2], lw=1.6, label="Fault detection rate")
ax.plot(thr, fpr, color=C[1], lw=1.6, label="False-alarm rate (Normal)")
ax.axvline(E["A"]["thr"], color=INK2, ls="--", lw=0.8); ax.text(E["A"]["thr"] + 0.2, 90, "calibrated (load 0)", fontsize=7)
ax.set_xlabel("Envelope SNR threshold (dB)"); ax.set_ylabel("%"); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig.savefig(FIG + "/threshold.png"); plt.close()
print("ok", sorted(os.listdir(FIG)))
