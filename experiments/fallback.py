"""Fault-injection study of the hybrid cloud/edge dispatcher against a mock LLM endpoint."""
import json, threading, time, urllib.request, urllib.error, socket, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

MODE = {"m": "ok"}
OK_DELAY = 0.35  # emulated cloud generation time (s) for the healthy path


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        m = MODE["m"]
        if m == "timeout":
            time.sleep(30); return
        if m in ("429", "404", "500"):
            return self._send(int(m), {"error": m})
        self._send(200, {"models": [{"name": "models/gemini-2.0-flash", "supportedGenerationMethods": ["generateContent"]}]})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        m = MODE["m"]
        if m == "timeout":
            time.sleep(30); return
        if m in ("429", "404", "500"):
            return self._send(int(m), {"error": m})
        if m == "malformed":
            return self._send(200, {"candidates": []})
        time.sleep(OK_DELAY)
        self._send(200, {"candidates": [{"content": {"parts": [{"text": "Diagnosis: outer race spall. Action: replace bearing."}]}}]})


def deterministic_report(label, kurt, bpfo, vel, rec):
    return (f"### AUTONOMOUS RELIABILITY ACTION DIRECTIVE (EDGE)\nDiagnosis: {label}; kurtosis {kurt:.2f}; "
            f"defect frequency {bpfo:.1f} Hz; velocity {vel:.2f} mm/s.\nPrecedent: {rec['record_id']} "
            f"({rec['failure_mechanism']}).\nAction: {rec['action']}.\nLOTO: {rec['loto_procedure']}.\n"
            f"Parts: {', '.join(rec['parts'])}.")


REC = dict(record_id="CMMS-WO-2019-1001", failure_mechanism="2.6 Fatigue", action="replace bearing",
           loto_procedure="LOTO-ELEC-480V", parts=["bearing"])


def call(url, data=None, timeout=12.0):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def naive_dispatch(base, list_timeout=6.0, gen_timeout=12.0):
    """Mirrors prototype server.py: model discovery, then sequential candidate models, then fallback."""
    t0 = time.perf_counter(); sel = None
    try:
        d = call(base + "/v1beta/models", timeout=list_timeout)
        ms = [m["name"].split("/")[-1] for m in d["models"]]
        sel = ms[0] if ms else None
    except Exception:
        pass
    cands = [sel] if sel else ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]
    body = json.dumps({"contents": [{"parts": [{"text": "diagnose"}]}]}).encode()
    for c in cands:
        try:
            r = call(f"{base}/v1beta/models/{c}:generateContent", body, timeout=gen_timeout)
            txt = r["candidates"][0]["content"]["parts"][0]["text"]
            return "CLOUD", txt, time.perf_counter() - t0
        except Exception:
            continue
    return "EDGE", deterministic_report("OR", 21.8, 105.9, 0.4, REC), time.perf_counter() - t0


class Breaker:
    """Circuit breaker with two request budgets, as shipped in backend/server.py.

    first_timeout is used while the dependency is healthy (a real LLM needs
    seconds to generate a report); the short probe timeout applies once a
    failure has been seen, so a hanging provider cannot hold the operator up on
    every subsequent request.
    """
    def __init__(self, fail_threshold=2, cooldown=30.0, timeout=2.0, first_timeout=15.0):
        self.ft, self.cd, self.to, self.first_to = fail_threshold, cooldown, timeout, first_timeout
        self.fails = 0; self.open_until = 0.0

    def budget(self):
        return self.first_to if self.fails == 0 else self.to

    def dispatch(self, base):
        t0 = time.perf_counter()
        if time.time() < self.open_until:
            return "EDGE", deterministic_report("OR", 21.8, 105.9, 0.4, REC), time.perf_counter() - t0
        body = json.dumps({"contents": [{"parts": [{"text": "diagnose"}]}]}).encode()
        try:
            r = call(f"{base}/v1beta/models/gemini-2.0-flash:generateContent", body, timeout=self.budget())
            txt = r["candidates"][0]["content"]["parts"][0]["text"]
            self.fails = 0
            return "CLOUD", txt, time.perf_counter() - t0
        except Exception:
            self.fails += 1
            if self.fails >= self.ft:
                self.open_until = time.time() + self.cd
            return "EDGE", deterministic_report("OR", 21.8, 105.9, 0.4, REC), time.perf_counter() - t0


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 18080), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:18080"
    dead = "http://127.0.0.1:18099"  # nothing listening -> connection refused (network partition)
    scen = [("Healthy (200 OK)", "ok", base), ("HTTP 429 rate limit", "429", base), ("HTTP 404 model not found", "404", base),
            ("HTTP 500 server error", "500", base), ("Malformed 200 response", "malformed", base),
            ("Connection refused", "ok", dead), ("Upstream timeout (hang)", "timeout", base)]
    out = []
    for name, m, url in scen:
        MODE["m"] = m
        for strat in ("naive", "breaker"):
            n = 3 if (m == "timeout" and strat == "naive") else 30
            br = Breaker()
            lat, paths, ok = [], [], 0
            for i in range(n):
                p, txt, dt = naive_dispatch(url) if strat == "naive" else br.dispatch(url)
                lat.append(dt * 1000); paths.append(p); ok += bool(txt)
            lat = np.array(lat)
            row = dict(scenario=name, strategy=strat, n=n, availability=ok / n,
                       cloud_frac=paths.count("CLOUD") / n, median_ms=float(np.median(lat)),
                       p95_ms=float(np.percentile(lat, 95)), max_ms=float(lat.max()), first_ms=float(lat[0]))
            out.append(row); print(row, flush=True)
    json.dump(out, open("results/fallback.json", "w"), indent=1)
    srv.shutdown()
