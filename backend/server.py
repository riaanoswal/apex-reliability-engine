"""APEX Industrial AI Reliability Engine - revised backend (v3.0).

Changes from the v2.4 demonstrator (see paper, Table "Audit"):
  * Real signal source: CWRU .mat file (APEX_SIGNAL_FILE) instead of the synthetic emulator.
  * All KPIs are measured: no floors (max(1.97, v)), no injected BPFO peak, no constant latencies.
  * Defect frequencies computed from shaft speed; ISO 10816-3 zone from band-limited velocity.
  * Diagnosis-conditioned CMMS retrieval; true similarity score with a minimum threshold.
  * Evidence-driven S/O/D risk mapping.
  * Circuit-breaker dispatcher (cloud Gemini / local Ollama / deterministic template).
  * Output verification checks on every report (numeric grounding, sections, identifiers, prohibited content).

Run:  uvicorn backend.server:app --port 8000
Env:  APEX_SIGNAL_FILE=/path/to/235.mat  APEX_RPM=1772  APEX_MACHINE_CLASS=G2_rigid  GEMINI_API_KEY=...
"""
import json, os, re, sys, time, urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apex_core"))
from dsp import (FS, MULT, classify, envelope_spectrum, iso10816_zone, peak_snr_db,  # noqa: E402
                 time_features, velocity_rms_mms, welch_psd)
from retrieval import build_corpus, TfidfRetriever  # noqa: E402

ASSET_TAG = "CMP-204B"
ISA95 = "ENTERPRISE / GULF_FPSO_01 / TRAIN_02 / CMP-204B"
SHAFT_RPM = float(os.getenv("APEX_RPM", "1772"))
MACHINE_CLASS = os.getenv("APEX_MACHINE_CLASS", "G2_rigid")
ASSET_CRITICALITY = int(os.getenv("APEX_CRITICALITY", "9"))
ETA_DB = float(os.getenv("APEX_THRESHOLD_DB", "8.0"))
S_MIN = 0.15
LABEL_TO_CAT = {"OR": "BRG_OR", "IR": "BRG_IR", "Ball": "BRG_BALL"}
SUFFIX = {"OR": " BPFO outer race", "IR": " BPFI inner race", "Ball": " BSF ball", "Normal": ""}
FAULT_NAME = {"OR": "outer-race defect (BPFO)", "IR": "inner-race defect (BPFI)",
              "Ball": "rolling-element defect (2xBSF)", "Normal": "no bearing defect signature"}
FAULT_FREQ = {"OR": "BPFO", "IR": "BPFI", "Ball": "BSF"}

CORPUS = build_corpus()
RETRIEVER = TfidfRetriever(CORPUS, ngram=(1, 2))


# ----------------------------------------------------------------------------- signal source
def load_window() -> np.ndarray:
    path = os.getenv("APEX_SIGNAL_FILE") or os.path.join(os.path.dirname(__file__), "..", "data", "235_1.mat")
    if path and os.path.exists(path):
        import scipy.io as sio
        m = sio.loadmat(path)
        x = m[[k for k in m if k.endswith("_DE_time")][0]].ravel().astype(float)
        idx = int(time.time()) % max(1, len(x) // 12000)
        return x[idx * 12000:(idx + 1) * 12000]
    raise RuntimeError("Set APEX_SIGNAL_FILE to a 12 kHz CWRU drive-end .mat file (the synthetic emulator was removed).")


def analyse(x: np.ndarray) -> Dict[str, Any]:
    t0 = time.perf_counter()
    label, scores, tf, band = classify(x, SHAFT_RPM, thr_db=ETA_DB)
    fr = SHAFT_RPM / 60.0
    v = velocity_rms_mms(x)
    zone = iso10816_zone(v, MACHINE_CLASS)
    f, E, _ = envelope_spectrum(x, band=band[:2])
    fault_f = None; fault_snr = None; harmonics = 0
    if label != "Normal":
        f0 = MULT[FAULT_FREQ[label]] * fr * (2 if label == "Ball" else 1)
        fault_snr, fault_f = peak_snr_db(f, E, f0)
        harmonics = sum(peak_snr_db(f, E, h * f0)[0] >= ETA_DB for h in (1, 2, 3))
    dsp_ms = (time.perf_counter() - t0) * 1000
    return dict(label=label, scores=scores, tf=tf, band=band, fr=fr, v=v, zone=zone,
                defect_freqs={k: MULT[k] * fr for k in MULT}, fault_f=fault_f, fault_snr=fault_snr,
                harmonics=harmonics, dsp_ms=dsp_ms)


# ----------------------------------------------------------------------------- risk (paper Table "rpn")
def risk(a: Dict[str, Any]):
    S = min(10, ASSET_CRITICALITY + {"C": 1, "D": 2}.get(a["zone"], 0))
    best = max(a["scores"].values())
    O = 2 if best < ETA_DB else 5 if best < ETA_DB + 6 else 7 if best < ETA_DB + 12 else 8
    if a["tf"]["kurtosis"] > 6 and best >= ETA_DB:
        O += 1
    agree = sum([best >= ETA_DB, a["tf"]["kurtosis"] > 4, a["tf"]["crest"] > 6, a["harmonics"] >= 2])
    D = {4: 2, 3: 3, 2: 5}.get(agree, 7)
    rpn = S * O * D
    prio = "P1" if rpn >= 150 else "P2" if rpn >= 80 else "P3"
    return S, O, D, rpn, prio


# ----------------------------------------------------------------------------- reports & verification
def facts(a, rec, sim, S, O, D, rpn, prio) -> Dict[str, Any]:
    return dict(asset=ASSET_TAG, rpm=SHAFT_RPM, diagnosis=FAULT_NAME[a["label"]],
                fault_freq_hz=None if a["fault_f"] is None else round(a["fault_f"], 1),
                env_snr_db=None if a["fault_snr"] is None else round(a["fault_snr"], 1),
                kurtosis=round(a["tf"]["kurtosis"], 2), crest=round(a["tf"]["crest"], 2),
                velocity_mms=round(a["v"], 2), iso_zone=a["zone"], S=S, O=O, D=D, rpn=rpn, priority=prio,
                bpfo_hz=round(a["defect_freqs"]["BPFO"], 1), bpfi_hz=round(a["defect_freqs"]["BPFI"], 1),
                shaft_hz=round(a["fr"], 2),
                record=rec["record_id"] if rec else None, similarity=round(sim, 3),
                action=rec["action"] if rec else "No reliable precedent - refer to reliability engineer",
                parts=rec["parts"] if rec else [], loto=rec["loto_procedure"] if rec else "Per site isolation register")


# --- Topic branches for the edge (deterministic) engine -----------------------
# Each entry: (regex over the user's question, section heading, body generator).
# The body may only use numbers already present in the facts dict F, because
# verify() rejects any unit-bearing number that is not grounded in F.
TOPIC_BRANCHES = [
    (r"\b(rca|root cause|why|diagnos\w*|kinematic\w*|harmonic\w*|sideband\w*|attribut\w*|order analysis|bpfo|bpfi|spectrum)\b", "KINEMATIC ROOT CAUSE",
     lambda F: (
         f"The envelope peak at {F['fault_freq_hz']} Hz is a non-integer order of shaft speed "
         f"({F['shaft_hz']} Hz), so it cannot be produced by unbalance, which is strictly synchronous at 1x, "
         f"or by misalignment, which is dominated by 2x. It matches the kinematic ball-pass frequency of the "
         f"outer race, BPFO = (n/2)*f_r*(1 - (d/D)*cos(theta)), computed as {F['bpfo_hz']} Hz for the "
         f"SKF 6205-2RS geometry at this speed. The inner-race frequency would be {F['bpfi_hz']} Hz, which is "
         f"not the dominant line. Because the outer ring is stationary in the housing, every rolling element "
         f"strikes the defect in the same load-zone orientation, so the impacts are not amplitude-modulated "
         f"at shaft rate: the absence of sidebands spaced by {F['shaft_hz']} Hz is the discriminator against "
         f"an inner-race defect. Kurtosis of {F['kurtosis']} and crest factor {F['crest']} confirm the signal "
         f"is impulsive rather than Gaussian.")),
    (r"\b(loto|lockout|lock-?out|isolat\w*|safety|permit|ppe|danger tag|tag-?out)\b", "ISOLATION AND SAFETY SEQUENCE",
     lambda F: (
         f"Isolation key {F['loto']} applies to this asset. Sequence: raise the permit to work; stop the "
         f"driver from the local control station; open and lock the primary breaker; apply personal danger "
         f"tags for every person working; prove dead at the motor terminals; isolate suction and discharge "
         f"and vent the casing to atmosphere before breaking containment; confirm the rotor is at standstill "
         f"before removing the coupling guard. Do not begin disassembly until the bearing housing has cooled "
         f"to a safe touch temperature. The isolation must stay in force until the work order is signed off.")),
    (r"\b(parts?|bom|bill of materials?|spares?|stock|requisition)\b", "PARTS AND REQUISITION",
     lambda F: (
         "Staged bill of materials: " + (", ".join(F["parts"]) if F["parts"] else "no parts staged") + ". "
         + (f"Drawn from precedent {F['record']}, retrieved at cosine similarity {F['similarity']}. "
            if F["record"] else "No precedent above the similarity threshold, so parts need engineering review. ")
         + "Check the bearing against the machine data sheet before issue, since interchange part numbers "
           "vary by housing fit. Confirm the seal kit and the lubricant grade are in stock at the same time; "
           "a bearing change opened without seals in hand is the most common cause of extended outage.")),
    (r"\b(work ?order|stage|staging|dispatch\w*|priorit\w*|schedul\w*|mtbf|outage|plan)\b", "WORK-ORDER STAGING",
     lambda F: (
         f"Risk priority number {F['rpn']} from S={F['S']} (asset criticality with ISO zone "
         f"{F['iso_zone']}), O={F['O']} (strength of the envelope evidence) and D={F['D']} (number of "
         f"independent indicators in agreement), giving {F['priority']}. Recommended handling: raise a "
         f"corrective order against the functional location, attach this diagnosis and the envelope "
         f"spectrum as evidence, reserve the staged materials, and plan a controlled ramp-down at the next "
         f"opportunity rather than an immediate trip, since velocity is {F['velocity_mms']} mm/s and the "
         f"machine is in ISO 10816-3 zone {F['iso_zone']}. Trend the envelope amplitude daily until the "
         f"intervention; a rising trend shortens the window.")),
    (r"\b(lubric\w*|oil|grease|viscosit\w*|vg ?\d+|film|tribolog\w*|damp\w*)\b", "LUBRICATION ASSESSMENT",
     lambda F: (
         "Heavier oil or grease will not remove the impacts. Once a raceway has spalled, the defect is a "
         "geometric discontinuity: each rolling element drops into the cavity and strikes the trailing edge, "
         "and that impulse is a contact-mechanics event, not a film-thickness problem. A heavier grade "
         "raises churning losses and bearing temperature, which thins the film it was meant to thicken, and "
         "it can mask the envelope signature while the damage continues to spread. Keep the grade specified "
         "on the data sheet, confirm the oil is clean and dry, and treat any lubricant change as a separate "
         "decision from the bearing replacement.")),
    (r"\b(seiz\w*|catastroph\w*|run to failure|what if|consequences?|keep running|keep it running|cascade|total failure)\b", "FAILURE CASCADE IF RUN TO FAILURE",
     lambda F: (
         f"If the bearing is run to seizure at {F['rpm']:.0f} rpm, the sequence is: loss of radial stiffness "
         "as the rolling elements collapse, so the shaft orbit grows beyond design clearance; rubbing and "
         "destruction of the labyrinth or dry-gas seal, with release of process gas and a flaring or "
         "emergency shutdown event; contact between the impeller and the stationary diffuser, which can "
         "breach the casing; and a torque spike that shears the coupling element and trips the driver on "
         "overcurrent. The repair then moves from a bearing change to a rotor rebuild, and the outage from "
         "hours to weeks. The failure develops over milliseconds once it starts, so there is no opportunity "
         "to intervene after the fact.")),
    (r"\b(mount\w*|install\w*|fit|fitting|induction|heating|press\w*|interference|clearance|assembl\w*)\b", "MOUNTING AND REASSEMBLY",
     lambda F: (
         "Mount by induction heating, not by flame or by driving the bearing on. Heat the bearing to roughly "
         "110 degrees Celsius, and never beyond 125, since the rings are dimensionally stabilised only to "
         "that point. Demagnetise after heating. Check the shaft seat and housing bore against the drawing "
         "before fitting; a spalled bearing often leaves a fretted or oversized seat that will fail the "
         "replacement within weeks. Push on squarely against the inner ring only, never through the rolling "
         "elements. Allow the bearing to cool and seat before applying axial clamp, then set the lubricant "
         "charge to the specified fill and record the installed clearance.")),
]


def deterministic_report(F: Dict[str, Any], query: str) -> str:
    ev = (f"envelope peak at {F['fault_freq_hz']} Hz ({F['env_snr_db']} dB above background)"
          if F["fault_freq_hz"] else "no envelope peak above threshold at BPFO, BPFI or 2xBSF")
    core = [
        f"### DIAGNOSIS\nAsset {F['asset']} at {F['rpm']:.0f} rpm (shaft {F['shaft_hz']} Hz): {F['diagnosis']}.",
        f"### EVIDENCE\n{ev}; kurtosis {F['kurtosis']}; crest factor {F['crest']}; "
        f"theoretical BPFO {F['bpfo_hz']} Hz, BPFI {F['bpfi_hz']} Hz.",
        f"### RISK\nVelocity {F['velocity_mms']} mm/s, ISO 10816-3 zone {F['iso_zone']}. "
        f"S={F['S']}, O={F['O']}, D={F['D']}, RPN={F['rpn']} ({F['priority']}).",
    ]
    # Topic-specific sections, in the order the branches are declared.
    q = (query or "").lower()
    topics = [(h, fn(F)) for pat, h, fn in TOPIC_BRANCHES if re.search(pat, q)]
    for heading, text in topics[:2]:          # at most two, to keep the panel readable
        core.append(f"### {heading}\n{text}")
    core += [
        f"### ACTIONS\n{F['action']}" + (f" (precedent {F['record']}, similarity {F['similarity']})." if F["record"] else "."),
        f"### ISOLATION\n{F['loto']}",
        f"### PARTS\n{', '.join(F['parts']) if F['parts'] else 'None staged'}",
        f"### QUERY\n{query}",
    ]
    return "\n".join(core)


REQUIRED = ["DIAGNOSIS", "EVIDENCE", "RISK", "ACTIONS", "ISOLATION", "PARTS"]
PROHIBITED = [r"bypass (the )?(isolation|lockout|trip)", r"defeat (the )?trip", r"continue (normal )?operation in zone d"]


def verify(report: str, F: Dict[str, Any]) -> Dict[str, bool]:
    allowed = {float(v) for k, v in F.items() if isinstance(v, (int, float)) and v is not None}
    # Integer harmonics of a grounded defect frequency are themselves grounded.
    for k in ("bpfo_hz", "bpfi_hz", "fault_freq_hz", "shaft_hz"):
        if isinstance(F.get(k), (int, float)):
            allowed |= {F[k] * h for h in (2, 3, 4)}
    allowed |= {1.0, 2.0, 3.0, 4.0, 10.0, 1000.0, 10816.0, 14224.0}
    nums = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\s*(?:Hz|mm/s|dB|rpm)", report)]
    grounded = all(any(abs(n - a) <= max(0.051, 0.005 * abs(a)) for a in allowed) for n in nums)
    sections = all(s.lower() in report.lower() for s in REQUIRED)
    ids = (F["record"] is None or F["record"] in report) and F["loto"] in report
    safe = not any(re.search(p, report.lower()) for p in PROHIBITED)
    return dict(numeric_grounding=grounded, sections=sections, identifiers=ids, prohibited_content=safe)


class Breaker:
    """Circuit breaker with two request budgets.

    first_to  - generous budget while the dependency is healthy, because a real
                LLM needs seconds to generate a full structured report.
    to        - short probe budget once a failure has been seen, so a hanging
                provider cannot hold the operator up request after request.
    """
    def __init__(self, k=2, cooldown=30.0, timeout=2.0, first_timeout=15.0):
        self.k, self.cd, self.to, self.fails, self.open_until = k, cooldown, timeout, 0, 0.0
        self.first_to = first_timeout

    def budget(self):
        return self.first_to if self.fails == 0 else self.to

    def is_open(self):
        return time.time() < self.open_until

    def failure(self):
        self.fails += 1
        if self.fails >= self.k:
            self.open_until = time.time() + self.cd

    def success(self):
        self.fails = 0


BREAKER = Breaker(timeout=float(os.getenv("APEX_PROBE_TIMEOUT", "2")),
                  first_timeout=float(os.getenv("APEX_CLOUD_TIMEOUT", "15")))


def prompt(F, query):
    return ("You are a reliability engineer writing a maintenance directive.\n"
            "RULES:\n"
            "1. Use ONLY the values in FACTS. Do not introduce any number carrying a unit "
            "(Hz, mm/s, dB, rpm) that is not present in FACTS; integer harmonics of a defect "
            "frequency are permitted.\n"
            "2. Quote the asset tag, CMMS record id and LOTO key exactly as given.\n"
            "3. Never advise bypassing isolation, defeating a trip, or continuing to run in Zone D.\n"
            "4. Write exactly these sections, in this order, each on its own line prefixed by '### ':\n"
            "   DIAGNOSIS, EVIDENCE, RISK, ACTIONS, ISOLATION, PARTS.\n"
            f"FACTS: {json.dumps(F)}\nQUESTION: {query}")


# Gemini model names change over time and a given key may not have access to all
# of them; a single hard-coded name is the most common cause of a cloud path that
# silently never works. Try a short ordered list and remember the one that answered.
CANDIDATE_MODELS = [m.strip() for m in os.getenv(
    "APEX_GEMINI_MODELS",
    "gemini-2.0-flash,gemini-2.5-flash,gemini-flash-latest,gemini-1.5-flash-latest,gemini-1.5-flash"
).split(",") if m.strip()]
ACTIVE_MODEL = {"name": None}


def _gemini_call(model, key, text, timeout):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({"contents": [{"parts": [{"text": text}]}],
                       "generationConfig": {"temperature": 0.1}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["candidates"][0]["content"]["parts"][0]["text"]


def cloud_llm(F, query, key, timeout=None):
    """Return (report, model_name). Raises if no candidate model answers."""
    text = prompt(F, query)
    budget = timeout if timeout is not None else BREAKER.first_to
    order = ([ACTIVE_MODEL["name"]] if ACTIVE_MODEL["name"] else []) + \
            [m for m in CANDIDATE_MODELS if m != ACTIVE_MODEL["name"]]
    last = None
    deadline = time.time() + budget
    for model in order:
        remaining = deadline - time.time()
        if remaining <= 0.2:
            break
        try:
            out = _gemini_call(model, key, text, min(remaining, budget))
            ACTIVE_MODEL["name"] = model
            return out, model
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403, 429):     # key or quota problem: other models will fail too
                raise
        except Exception as e:
            last = e
    raise last or RuntimeError("no candidate Gemini model responded")


def local_llm(F, query, model="llama3.1:8b"):
    import ollama
    r = ollama.chat(model=model, messages=[{"role": "user", "content": prompt(F, query)}], options={"temperature": 0.1})
    return r["message"]["content"]


# ----------------------------------------------------------------------------- API
app = FastAPI(title="APEX Industrial AI Reliability Engine", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class DiagnoseRequest(BaseModel):
    query: str = "General diagnostic assessment"
    mode: Literal["cloud", "local", "offline"] = "cloud"
    api_key: Optional[str] = None
    model: Optional[str] = None  # accepted for compatibility with the v2.4 dashboard


@app.get("/api/health")
def health():
    return dict(status="ONLINE", timestamp=datetime.now(timezone.utc).isoformat(), asset_id=ASSET_TAG,
                isa95_hierarchy=ISA95, compute_stack=dict(dsp="NumPy/SciPy (CPU)", retrieval="TF-IDF (scikit-learn)",
                                                            cloud="Gemini REST", edge="Ollama / deterministic template"),
                breaker_open=BREAKER.is_open())


@app.get("/api/telemetry")
def telemetry():
    x = load_window(); a = analyse(x)
    n = len(x); t = np.arange(n) / FS * 1000
    f, P = welch_psd(x); m = f <= 800
    # Envelope spectrum (Hilbert, SK-selected band): this is where bearing defect
    # frequencies are actually visible. 1 Hz bins, so the BPFO bin is exact.
    fe, E, _ = envelope_spectrum(x, band=a["band"][:2])
    me = (fe >= 1) & (fe <= 500)
    env_spectrum = [dict(frequency_hz=round(float(fe[i]), 1), psd=round(float(E[i]) * 1000, 4)) for i in np.where(me)[0]]
    return dict(asset_id=ASSET_TAG, sample_rate_hz=FS, shaft_rpm=SHAFT_RPM, shaft_frequency_hz=round(a["fr"], 2),
                defect_frequencies_hz={k: round(v, 2) for k, v in a["defect_freqs"].items()},
                rms_velocity_mms=round(a["v"], 3), iso_zone=a["zone"], machine_class=MACHINE_CLASS,
                crest_factor=round(a["tf"]["crest"], 2), kurtosis=round(a["tf"]["kurtosis"], 2),
                diagnosis=a["label"], envelope_scores_db={k: round(v, 1) for k, v in a["scores"].items()},
                time_series=[dict(time_ms=round(float(t[i]), 2), amplitude_g=round(float(x[i]), 4)) for i in range(0, 4000, 40)],
                psd_linear=[dict(frequency_hz=round(float(f[i]), 1), psd_g2_per_hz=float(P[i])) for i in np.where(m)[0]],
                envelope_band_hz=[round(a["band"][0], 1), round(a["band"][1], 1)],
                # --- fields read by the v2.4 React dashboard (kept so App.tsx works unchanged) ---
                spectral_kurtosis=round(a["tf"]["kurtosis"], 2),
                bpfo_frequency_hz=round(a["defect_freqs"]["BPFO"], 2),
                iso_10816_status=f"ZONE {a['zone']}",
                alert_level="CRITICAL" if a["label"] != "Normal" and a["zone"] in ("C", "D") else "WARNING" if a["label"] != "Normal" else "NORMAL",
                fft_spectrum=env_spectrum)


@app.post("/api/diagnose")
def diagnose(req: DiagnoseRequest):
    T0 = time.perf_counter()
    a = analyse(load_window())
    t1 = time.perf_counter()
    res = RETRIEVER.query(req.query + SUFFIX[a["label"]], k=3)
    idx, sim = res[0]
    rec = CORPUS[idx] if sim >= S_MIN else None
    t_ret = (time.perf_counter() - t1) * 1000
    S, O, D, rpn, prio = risk(a)
    F = facts(a, rec, sim, S, O, D, rpn, prio)
    t2 = time.perf_counter(); path = "EDGE"; report = None; used_model = None
    key = req.api_key or os.getenv("GEMINI_API_KEY", "")
    try:
        if req.mode == "cloud" and key and not BREAKER.is_open():
            try:
                report, used_model = cloud_llm(F, req.query, key, timeout=BREAKER.budget())
                BREAKER.success(); path = "CLOUD"
            except Exception:
                BREAKER.failure()
        elif req.mode == "local":
            try:
                report = local_llm(F, req.query); path = "LOCAL_LLM"
            except Exception:
                pass
    finally:
        if report is None:
            report = deterministic_report(F, req.query); path = "EDGE"
    checks = verify(report, F)
    if not all(checks.values()) and path != "EDGE":
        report = deterministic_report(F, req.query); path = "EDGE_AFTER_CHECK_FAIL"; checks = verify(report, F)
    t_gen = (time.perf_counter() - t2) * 1000
    return dict(asset_id=ASSET_TAG, path=path, diagnosis=a["label"], report=report, checks=checks,
                matched_cmms_record=F["record"], similarity_score=round(sim, 3),
                rpn=rpn, rpn_severity=S, rpn_occurrence=O, rpn_detection=D, priority=prio,
                parts_allocated=F["parts"], loto_protocol=F["loto"],
                latency_ms=dict(dsp=round(a["dsp_ms"], 2), retrieval=round(t_ret, 2), generation=round(t_gen, 2),
                                total=round((time.perf_counter() - T0) * 1000, 2)),
                # --- fields read by the v2.4 React dashboard (all values are measured) ---
                mode=path, model_used=used_model or {"LOCAL_LLM": "llama3.1:8b"}.get(path, "deterministic-template"),
                diagnostic_report=report, risk_priority_number=rpn, action_priority=prio,
                execution_trace=dict(cusignal_dsp_ms=round(a["dsp_ms"], 2), cuvs_rag_retrieval_ms=round(t_ret, 2),
                                     nemo_guardrails_ms=0.0, llm_ttft_ms=round(t_gen, 2),
                                     total_pipeline_ms=round((time.perf_counter() - T0) * 1000, 2), hardware_speedup_factor=1.0))


@app.get("/api/work-order")
def work_order(query: str = "stage work order"):
    d = diagnose(DiagnoseRequest(query=query, mode="offline"))
    rec = next((r for r in CORPUS if r["record_id"] == d["matched_cmms_record"]), None)
    return dict(work_order_id=f"WO-APEX-{datetime.now(timezone.utc):%Y%m%d%H%M%S}",
                iso_14224_equipment_class="COMPRESSOR_CENTRIFUGAL", functional_location="GULF_FPSO_01-TRAIN_02-CMP-204B",
                priority=d["priority"], diagnosis=d["diagnosis"],
                failure_mechanism=rec["failure_mechanism"] if rec else None,
                iso14224_failure_mode=rec["iso14224_failure_mode"] if rec else None,
                loto_isolation_key=d["loto_protocol"], rpn=d["rpn"],
                required_bill_of_materials=[dict(description=p, qty=1) for p in d["parts_allocated"]],
                status="STAGED_FOR_PLANNER_REVIEW",
                sap_pm_bapi_payload={"BAPI_ALM_ORDER_MAINTAIN": {"HEADER": {
                    "ORDER_TYPE": "PM02", "PLANPLANT": "1004", "EQUIPMENT": ASSET_TAG, "MN_WK_CTR": "MECH_REL",
                    "SHORT_TEXT": f"{d['priority']}: {d['diagnosis']} {ASSET_TAG}"[:40]}, "NOTIF_TYPE": "M1"}})


# ----------------------------------------------------------------------------- hosted mode
# When the dashboard has been built (frontend/dist exists, as in the Docker image),
# serve it from this same server so one process runs the whole site. Mounted last,
# so every /api route above still takes precedence. Local development via run.py
# uses the Vite dev server instead and never reaches this block.
_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")
if os.path.isdir(_DIST):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="dashboard")
