"""APEX DSP core: real implementation used for all experiments in the paper."""
import numpy as np
from scipy import signal

FS = 12000.0
# CWRU drive-end bearing SKF 6205-2RS JEM, defect frequency multiples of shaft speed
MULT = {"BPFI": 5.4152, "BPFO": 3.5848, "FTF": 0.39828, "BSF": 4.7135}
NOMINAL_RPM = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}


def time_features(x):
    xc = x - x.mean()
    rms = float(np.sqrt(np.mean(xc ** 2)))
    peak = float(np.max(np.abs(xc)))
    kurt = float(np.mean(xc ** 4) / (np.mean(xc ** 2) ** 2))
    skew = float(np.mean(xc ** 3) / (np.mean(xc ** 2) ** 1.5))
    return {"rms_g": rms, "peak_g": peak, "crest": peak / rms, "kurtosis": kurt, "skew": skew}


def velocity_rms_mms(x, fs=FS, f_lo=10.0, f_hi=1000.0):
    """Frequency-domain integration of acceleration (g) to velocity (mm/s RMS), ISO 10816 band."""
    n = len(x)
    X = np.fft.rfft((x - x.mean()) * 9.80665)
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f >= f_lo) & (f <= f_hi)
    V = np.zeros_like(X)
    V[band] = X[band] / (1j * 2 * np.pi * f[band])
    v = np.fft.irfft(V, n) * 1000.0
    return float(np.sqrt(np.mean(v ** 2)))


def iso10816_zone(v, group="G2_rigid"):
    limits = {"G2_rigid": (1.4, 2.8, 4.5), "G2_flex": (2.3, 4.5, 7.1),
              "G1_rigid": (2.3, 4.5, 7.1), "G1_flex": (3.5, 7.1, 11.0)}[group]
    return "A" if v < limits[0] else "B" if v < limits[1] else "C" if v < limits[2] else "D"


def welch_psd(x, fs=FS, nperseg=4096):
    return signal.welch(x - x.mean(), fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann")


def stft_spectral_kurtosis(x, fs=FS, nperseg=256):
    f, t, Z = signal.stft(x - x.mean(), fs=fs, nperseg=nperseg, noverlap=nperseg * 3 // 4, window="hann")
    P = np.abs(Z) ** 2
    sk = np.mean(P ** 2, axis=1) / (np.mean(P, axis=1) ** 2 + 1e-20) - 2.0
    return f, sk


def select_band(x, fs=FS, bw=None):
    """Pick demodulation band centred on max spectral kurtosis (simplified kurtogram)."""
    f, sk = stft_spectral_kurtosis(x, fs)
    valid = (f > 500) & (f < fs / 2 - 500)
    sk_s = np.convolve(sk, np.ones(5) / 5, mode="same")
    fc = float(f[valid][np.argmax(sk_s[valid])])
    bw = bw or 1000.0
    lo, hi = max(fc - bw / 2, 200.0), min(fc + bw / 2, fs / 2 - 100)
    return lo, hi, fc, f, sk


def envelope_spectrum(x, fs=FS, band=None):
    if band is None:
        lo, hi, fc, _, _ = select_band(x, fs)
    else:
        lo, hi = band
        fc = (lo + hi) / 2
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x - x.mean())
    env = np.abs(signal.hilbert(y))
    env = env - env.mean()
    n = len(env)
    E = np.abs(np.fft.rfft(env * np.hanning(n))) / n
    f = np.fft.rfftfreq(n, 1 / fs)
    return f, E, (lo, hi, fc)


def peak_snr_db(f, E, f0, tol=0.02, bg_halfwidth=10.0):
    """Max amplitude within +/-tol of f0 relative to local median background, dB."""
    win = (f >= f0 * (1 - tol)) & (f <= f0 * (1 + tol))
    bg = (f >= f0 - bg_halfwidth) & (f <= f0 + bg_halfwidth) & ~win
    if not win.any() or not bg.any():
        return 0.0, f0
    i = np.argmax(E[win])
    pk = E[win][i]
    return float(20 * np.log10(pk / (np.median(E[bg]) + 1e-15))), float(f[win][i])


def fault_scores(f, E, fr, harmonics=3):
    """Harmonic-averaged envelope SNR at each kinematic defect frequency."""
    scores = {}
    for name in ("BPFO", "BPFI", "BSF"):
        f0 = MULT[name] * fr
        if name == "BSF":
            f0 = 2 * f0  # ball defects strike both races: 2xBSF dominant
        snrs = [peak_snr_db(f, E, k * f0)[0] for k in range(1, harmonics + 1)]
        scores[name] = float(np.mean(snrs))
    return scores


def classify(x, rpm, thr_db=8.0, kurt_gate=None):
    """Deterministic kinematic rule engine (edge fallback path)."""
    fr = rpm / 60.0
    f, E, band = envelope_spectrum(x)
    sc = fault_scores(f, E, fr)
    best = max(sc, key=sc.get)
    tf = time_features(x)
    if sc[best] < thr_db:
        return "Normal", sc, tf, band
    return {"BPFO": "OR", "BPFI": "IR", "BSF": "Ball"}[best], sc, tf, band
