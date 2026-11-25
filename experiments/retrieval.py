"""ISO 14224-style synthetic CMMS corpus + TF-IDF / BM25 retrievers + retrieval evaluation."""
import json, math, re, collections, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# (category, ISO14224 failure mode, failure mechanism, component, symptom variants, actions, parts, LOTO)
CATS = {
 "BRG_OR": ("VIB", "2.6 Fatigue", "Drive-end rolling element bearing outer raceway",
   ["non-synchronous envelope peak at BPFO with harmonics, no shaft-rate sidebands, kurtosis above 6",
    "impulsive acceleration bursts at outer race defect frequency, high crest factor, spall in load zone",
    "demodulated spectrum shows ball pass frequency outer race series, rising high-frequency energy",
    "periodic impacts at 3.58x running speed, outer ring spalling found at inspection"],
   "replace drive-end bearing, inspect housing fit, flush and renew lubricant", ["SKF 6205-2RS / 22220-E bearing", "ISO VG 46 oil"], "LOTO-ELEC-480V"),
 "BRG_IR": ("VIB", "2.6 Fatigue", "Rolling element bearing inner raceway",
   ["envelope peak at BPFI with 1x shaft-rate sidebands, amplitude modulation from rotating load",
    "inner ring spall, impacts modulated by running speed, kurtosis elevated",
    "ball pass frequency inner race and harmonics spaced by 1x, spectral kurtosis band near 3 kHz",
    "5.4x running speed non-synchronous series with sidebands, flaking on inner race"],
   "replace bearing, check shaft seat and interference fit", ["Bearing assembly", "Shaft sleeve"], "LOTO-ELEC-480V"),
 "BRG_BALL": ("NOI", "2.4 Wear", "Rolling element (ball) of bearing",
   ["2x ball spin frequency in envelope, modulated at cage frequency, intermittent impacts",
    "rolling element pitting, BSF harmonics with FTF sidebands, erratic kurtosis",
    "ball defect, weak and intermittent envelope signature, noise from bearing",
    "ball spin frequency series observed, ball surface damage at teardown"],
   "replace bearing, analyse lubricant for debris", ["Bearing assembly"], "LOTO-ELEC-480V"),
 "BRG_CAGE": ("NOI", "2.5 Breakage", "Bearing cage / retainer",
   ["sub-synchronous fundamental train frequency around 0.4x, cage instability",
    "FTF peak and sidebands, rubbing noise, cage pocket wear",
    "fundamental train frequency modulation, fractured retainer",
    "low frequency 0.4x component, bearing noise, cage wear debris in oil"],
   "replace bearing and verify lubrication supply", ["Bearing assembly", "Oil filter"], "LOTO-ELEC-480V"),
 "UNBAL": ("VIB", "1.2 Vibration", "Rotor / impeller",
   ["dominant 1x running speed radial vibration, phase stable, low kurtosis near 3",
    "synchronous 1x peak rising after fouling of impeller, sinusoidal waveform",
    "once-per-revolution vibration, radial, in phase, deposit build-up on wheel",
    "high 1x amplitude, no harmonics, mass unbalance after blade erosion"],
   "clean impeller, perform in-situ two-plane balancing", ["Balancing weights"], "LOTO-MECH-GUARD"),
 "MISAL": ("VIB", "1.3 Clearance/alignment failure", "Flexible coupling / shaft line",
   ["dominant 2x running speed with high axial vibration, 180 degree phase across coupling",
    "angular and parallel misalignment, axial 1x and 2x, coupling heat",
    "2x synchronous harmonic, axial levels elevated, thermal growth offset",
    "coupling misalignment after maintenance, 2x and 3x harmonics"],
   "laser align driver train, re-shim, torque coupling bolts", ["Shim pack", "Coupling hardware kit"], "LOTO-MECH-DISC"),
 "LOOSE": ("VIB", "1.5 Looseness", "Baseplate / bearing housing",
   ["many harmonics of running speed, half-order subharmonics, truncated waveform",
    "mechanical looseness, 0.5x and 1x to 10x harmonic comb, directional vibration",
    "soft foot, anchor bolt loosening, rattling, harmonic series",
    "bearing housing looseness, raised floor noise, non-linear response"],
   "re-torque hold-down bolts, correct soft foot, grout repair", ["Anchor bolt kit", "Epoxy grout"], "LOTO-MECH-GUARD"),
 "SURGE": ("ERO", "4.0 Control failure", "Anti-surge valve / impeller stage",
   ["flow reversal, discharge pressure oscillation, sub-synchronous broadband at 0.4x to 0.6x",
    "compressor surge cycles, temperature rise at suction, anti-surge valve hunting",
    "rotating stall, low-frequency pressure pulsation, erratic output",
    "surge line crossed, IGV mis-set, acoustic resonance"],
   "recalibrate anti-surge controller and valve actuator, adjust IGV", ["Actuator diaphragm kit", "IGV bushing set"], "LOTO-GAS-ESD"),
 "SEAL": ("ELP", "1.1 Leakage", "Dry gas seal / labyrinth seal",
   ["primary vent flow high, seal gas differential pressure low, process gas leak",
    "dry gas seal leakage alarm, face damage, contamination of seal gas",
    "external leakage of process medium at shaft end seal",
    "labyrinth rub, increased leakage, seal clearance opened"],
   "replace seal cartridge, inspect seal gas filtration", ["Dry gas seal cartridge", "Seal gas filter"], "LOTO-GAS-ESD"),
 "LUBE": ("OHE", "2.7 Overheating", "Lube oil system / journal bearing",
   ["bearing metal temperature rising, oil viscosity loss, varnish",
    "overheating of journal bearing, lube oil degraded, water contamination",
    "high oil outlet temperature, cooler fouled, low lube pressure",
    "babbitt wipe, oil oxidation, particle count high"],
   "change oil, clean cooler, replace filters, check oil pump", ["Lube oil ISO VG 46", "Oil filter element"], "LOTO-ELEC-PUMP"),
 "COUPL": ("BRD", "2.4 Wear", "Gear / disc coupling",
   ["coupling disc pack cracks, torsional vibration, intermittent knocking",
    "gear coupling tooth wear, lubrication loss, axial locking",
    "coupling element failure, 1x and 2x with torsional content",
    "disc pack fatigue, bolt fretting, coupling guard hot"],
   "replace disc pack, inspect hub bores", ["Disc pack", "Coupling bolts M16"], "LOTO-MECH-DISC"),
 "MOTOR": ("PDE", "3.0 Electrical failure", "Induction motor rotor / stator",
   ["broken rotor bar, pole-pass sidebands around 1x, current signature sidebands",
    "stator winding insulation degradation, 2x line frequency vibration, hot spot",
    "motor electrical fault, twice line frequency 100 Hz or 120 Hz peak",
    "rotor bar crack, slip frequency sidebands, current unbalance"],
   "motor current signature analysis, rewind or replace rotor", ["Motor rotor", "Insulation kit"], "LOTO-ELEC-480V"),
}

QUERIES = {
 "BRG_OR": ["outer ring spall suspected on drive end, what did we do last time",
            "envelope shows BPFO harmonics without sidebands", "stationary race defect impacts in load zone",
            "bearing outer race damage history", "3.58 times shaft speed impacts found"],
 "BRG_IR": ["inner ring flaking with running speed sidebands", "BPFI series detected in demodulated spectrum",
            "rotating race defect modulated by shaft", "inner raceway spall repair record", "5.4x non synchronous peaks with 1x spacing"],
 "BRG_BALL": ["rolling element pitting found", "BSF peaks in envelope with cage modulation",
              "ball damage intermittent signature", "ball spin frequency detected", "bearing ball surface defect previous work order"],
 "BRG_CAGE": ["retainer broken", "cage frequency around 0.4 times speed", "FTF component in spectrum",
              "cage wear found in oil", "bearing retainer instability"],
 "UNBAL": ["strong once per rev vibration after fouling", "1x radial high, phase steady",
           "impeller deposit causing vibration", "need balancing job", "mass unbalance on rotor"],
 "MISAL": ["axial vibration high at twice running speed", "coupling out of line after overhaul",
           "2x harmonic with phase flip across coupling", "alignment problem driver", "shaft misalignment history"],
 "LOOSE": ["base bolts loose, rattling", "half order subharmonics and many harmonics",
           "soft foot on skid", "housing looseness", "truncated waveform harmonic comb"],
 "SURGE": ["discharge pressure swinging and flow reversal", "anti surge valve hunting",
           "compressor surging at low flow", "stall pulsation low frequency", "IGV setting wrong erratic flow"],
 "SEAL": ["gas leaking at shaft end", "seal vent flow alarm", "dry gas seal failure",
          "process gas leakage external", "seal face damaged"],
 "LUBE": ["journal bearing temperature climbing", "oil degraded and hot",
          "lube cooler fouled high oil temperature", "babbitt wiped", "water in lube oil"],
 "COUPL": ["disc pack cracked", "coupling knocking torsional", "gear coupling teeth worn",
           "coupling element broken", "coupling bolts fretting"],
 "MOTOR": ["broken rotor bar suspected", "twice line frequency vibration on motor",
           "motor current sidebands", "stator winding hot", "electrical fault in driver motor"],
}


def build_corpus():
    docs = []
    wo = 1000
    for cat, (fm, mech, comp, symptoms, action, parts, loto) in CATS.items():
        for i, s in enumerate(symptoms):
            wo += 1
            docs.append(dict(record_id=f"CMMS-WO-{2019 + i}-{wo}", category=cat, iso14224_failure_mode=fm,
                             failure_mechanism=mech, component=comp, symptoms=s, action=action, parts=parts,
                             loto_procedure=loto, text=f"{comp}. {fm} {mech}. {s}. {action}"))
    return docs


class TfidfRetriever:
    def __init__(self, docs, ngram=(1, 1)):
        self.docs = docs
        self.v = TfidfVectorizer(ngram_range=ngram, sublinear_tf=False)
        self.M = self.v.fit_transform([d["text"] for d in docs])

    def query(self, q, k=5):
        s = cosine_similarity(self.v.transform([q]), self.M).ravel()
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]


def tok(t):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t.lower())


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = docs; self.k1 = k1; self.b = b
        self.T = [tok(d["text"]) for d in docs]
        self.avg = np.mean([len(t) for t in self.T])
        self.df = collections.Counter(w for t in self.T for w in set(t))
        self.N = len(docs)
        self.tf = [collections.Counter(t) for t in self.T]

    def query(self, q, k=5):
        qs = tok(q); sc = np.zeros(self.N)
        for i, tf in enumerate(self.tf):
            L = len(self.T[i])
            for w in qs:
                if w in tf:
                    idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    sc[i] += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * L / self.avg))
        idx = np.argsort(-sc)[:k]
        return [(int(i), float(sc[i])) for i in idx]


STATIC_AUG = " BPFO 105.7Hz outer race bearing spalling high kurtosis"  # prototype server.py behaviour
DSP_AUG = {"BRG_OR": " BPFO outer race", "BRG_IR": " BPFI inner race", "BRG_BALL": " BSF ball",
           "BRG_CAGE": " FTF cage"}


def evaluate(retr, aug="none"):
    top1 = top3 = 0; rr = []; n = 0; to_or = 0
    for cat, qs in QUERIES.items():
        for q in qs:
            qq = q
            if aug == "static":
                qq = q + STATIC_AUG
            elif aug == "dsp":
                qq = q + DSP_AUG.get(cat, "")
            res = retr.query(qq, k=len(retr.docs))
            cats = [retr.docs[i]["category"] for i, _ in res]
            n += 1
            top1 += cats[0] == cat; top3 += cat in cats[:3]
            rr.append(1.0 / (cats.index(cat) + 1)); to_or += cats[0] == "BRG_OR"
    return dict(top1=top1 / n, top3=top3 / n, mrr=float(np.mean(rr)), frac_top1_BRG_OR=to_or / n, n=n)


if __name__ == "__main__":
    docs = build_corpus()
    out = {"n_docs": len(docs), "n_queries": sum(len(v) for v in QUERIES.values()), "n_categories": len(CATS)}
    for name, R in [("tfidf_uni", TfidfRetriever(docs, (1, 1))), ("tfidf_unibi", TfidfRetriever(docs, (1, 2))), ("bm25", BM25(docs))]:
        for aug in ("none", "static", "dsp"):
            out[f"{name}|{aug}"] = evaluate(R, aug)
            print(name, aug, {k: round(v, 3) for k, v in out[f"{name}|{aug}"].items()})
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/retrieval.json", "w"), indent=1)
