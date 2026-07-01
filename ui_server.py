from __future__ import annotations
import argparse
import csv
import datetime as dt
import glob
import json
import os
from flask import Flask, jsonify, request, Response

import config as C

RED_K = 0.15        
YELLOW_K = 0.3333   

BLEND_KNN_WEIGHT = 0.50
HAZARD_CLASSES = [c for c in C.ALL_LABELS if c != C.NONE_LABEL]

app = Flask(__name__)
app.config["LOG_PATHS"] = {}
app.config["REFS_CSV"] = None
app.config["EMB_PATHS"] = {}        
app.config["THRESH_PATH"] = None    


def load_static_file(relative_path):
    full_path = os.path.join("static", relative_path)
    with open(full_path, "r", encoding="utf-8") as file:
        return file.read()


def _default_thresholds():
    out = {}
    for c in HAZARD_CLASSES:
        w1, r1 = RED_K, round(RED_K * 2, 3)
        w24, r24 = YELLOW_K, round(YELLOW_K * 1.6, 3)
        out[c] = {"h1": {"watch": w1, "advisory": round((w1 + r1) / 2, 3), "warn": r1},
                  "h24": {"watch": w24, "advisory": round((w24 + r24) / 2, 3), "warn": r24}}
    return out


def load_thresholds():
    p = app.config["THRESH_PATH"]
    base = _default_thresholds()
    if p and os.path.exists(p):
        try:
            with open(p) as f:
                saved = json.load(f)
            for c in base:                      
                if c in saved:
                    for h in ("h1", "h24"):
                        if h in saved[c]:
                            base[c][h].update(saved[c][h])
        except Exception:
            pass
    return base


def save_thresholds(th):
    p = app.config["THRESH_PATH"]
    if not p:
        return False
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(th, f, indent=2)
    os.replace(tmp, p)
    return True


def _read_log(path):
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _location_key(path):
    recs = _read_log(path)
    if recs:
        return recs[-1].get("station") or os.path.splitext(os.path.basename(path))[0]
    return os.path.splitext(os.path.basename(path))[0]


def _resolve_loc():
    keys = list(app.config["LOG_PATHS"].keys())
    if request.method == "POST":
        loc = (request.get_json(silent=True) or {}).get("loc") or request.args.get("loc")
    else:
        loc = request.args.get("loc")
    loc = loc or (keys[0] if keys else None)
    return loc if loc in app.config["LOG_PATHS"] else None


def _read_refs():
    path = app.config["REFS_CSV"]
    if not path or not os.path.exists(path):
        return [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return rows, (r.fieldnames or [])


def _write_refs(rows, header):
    path = app.config["REFS_CSV"]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


@app.route("/api/locations")
def api_locations():
    out = []
    for key, path in app.config["LOG_PATHS"].items():
        recs = _read_log(path)
        latest = recs[-1] if recs else {}
        out.append({"key": key, "location": latest.get("location", key),
                    "station": latest.get("station", key), "n_records": len(recs),
                    "latest_utc": latest.get("data_latest_utc")})
    out.sort(key=lambda d: d["location"])
    return jsonify(out)


@app.route("/api/latest")
def api_latest():
    loc = _resolve_loc()
    if loc is None:
        return jsonify({"error": "unknown location"}), 404
    recs = _read_log(app.config["LOG_PATHS"][loc])
    if not recs:
        return jsonify({"error": "no records yet"}), 404
    return jsonify(recs[-1])


@app.route("/api/history")
def api_history():
    loc = _resolve_loc()
    if loc is None:
        return jsonify({"error": "unknown location"}), 404
    return jsonify(_read_log(app.config["LOG_PATHS"][loc]))


@app.route("/api/meta")
def api_meta():
    return jsonify({"classes": list(C.ALL_LABELS), "hazard_classes": HAZARD_CLASSES,
                    "none_label": C.NONE_LABEL, "red_k": RED_K, "yellow_k": YELLOW_K,
                    "blend_knn_weight": BLEND_KNN_WEIGHT, "has_refs": bool(app.config["REFS_CSV"]),
                    "has_thresholds": bool(app.config["THRESH_PATH"])})


@app.route("/api/thresholds")
def api_thresholds_get():
    return jsonify(load_thresholds())


@app.route("/api/thresholds/set", methods=["POST"])
def api_thresholds_set():
    if not app.config["THRESH_PATH"]:
        return jsonify({"error": "no thresholds.json configured (start with --thresholds)"}), 400
    body = request.get_json(force=True)
    cls, horizon, tier = body.get("class"), body.get("horizon"), body.get("tier")
    val = body.get("value")
    if cls not in HAZARD_CLASSES or horizon not in ("h1", "h24") or tier not in ("watch", "advisory", "warn"):
        return jsonify({"error": "bad class/horizon/tier"}), 400
    try:
        val = max(0.0, float(val))
    except (TypeError, ValueError):
        return jsonify({"error": "bad value"}), 400
    th = load_thresholds()
    row = th[cls][horizon]
    row.setdefault("advisory", round((row["watch"] + row["warn"]) / 2, 4))
    row[tier] = round(val, 4)
    
    lo, hi = min(row["watch"], row["warn"]), max(row["watch"], row["warn"])
    row["advisory"] = min(max(row["advisory"], lo), hi)
    save_thresholds(th)
    return jsonify({"ok": True, "class": cls, "horizon": horizon, "thresholds": th[cls]})


def _lookup_embedding(loc, data_latest_utc):

    emb_path = app.config["EMB_PATHS"].get(loc)
    if not emb_path or not os.path.exists(emb_path):
        return None
    recs = _read_log(app.config["LOG_PATHS"].get(loc))
    station = recs[-1].get("station") if recs else loc
    found = None
    with open(emb_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("station") == station and r.get("data_latest_utc") == data_latest_utc:
                found = r.get("embedding")
    return found


@app.route("/api/knn/refs")
def api_knn_refs():
    rows, _ = _read_refs()
    labels = list(C.ALL_LABELS)
    counts = {lab: 0 for lab in labels}
    dim = 0
    for row in rows:
        try:
            ci = int(float(row.get("class", -1)))
        except (TypeError, ValueError):
            continue
        if 0 <= ci < len(labels):
            counts[labels[ci]] += 1
        if not dim:
            dim = sum(1 for k in row if k.startswith("emb_"))
    return jsonify({"dim": dim, "total": len(rows), "counts": counts,
                    "path": app.config["REFS_CSV"]})


@app.route("/api/knn/records")
def api_knn_records():
    loc = _resolve_loc()
    if loc is None:
        return jsonify([])
    recs = _read_log(app.config["LOG_PATHS"][loc])
    emb_path = app.config["EMB_PATHS"].get(loc)
    
    have = set()
    if emb_path and os.path.exists(emb_path):
        with open(emb_path) as f:
            for line in f:
                try:
                    have.add(json.loads(line).get("data_latest_utc"))
                except Exception:
                    pass
    out = [{"idx": i, "data_latest_utc": r.get("data_latest_utc"),
            "has_emb": r.get("data_latest_utc") in have} for i, r in enumerate(recs)]
    return jsonify(out[::-1])           


@app.route("/api/knn/add", methods=["POST"])
def api_knn_add():
    body = request.get_json(force=True)
    loc = body.get("loc")
    if loc not in app.config["LOG_PATHS"]:
        return jsonify({"error": "unknown location"}), 400
    if not app.config["REFS_CSV"]:
        return jsonify({"error": "no refs CSV configured (start with --refs-csv)"}), 400
    recs = _read_log(app.config["LOG_PATHS"][loc])
    idx = int(body.get("record_idx", len(recs) - 1))
    if not (0 <= idx < len(recs)):
        return jsonify({"error": "bad record index"}), 400
    dlu = recs[idx].get("data_latest_utc")
    emb = _lookup_embedding(loc, dlu)
    if not emb:
        return jsonify({"error": "no embedding stored for that window"}), 400
    labels = list(C.ALL_LABELS)
    class_idxs = [labels.index(c) for c in body.get("classes", []) if c in labels]
    if not class_idxs:
        return jsonify({"error": "no valid classes given"}), 400

    rows, header = _read_refs()
    if not header:
        header = [f"emb_{i}" for i in range(len(emb))] + ["class"] + \
                 [f"prior_{c}" for c in range(len(labels))] + ["added_utc", "source"]
    for extra in ("added_utc", "source"):     
        if extra not in header:
            header = header + [extra]
    priors = {f"prior_{c}": (rows[0].get(f"prior_{c}", "") if rows else "")
              for c in range(len(labels))}
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    added = 0
    for ci in class_idxs:
        row = {f"emb_{i}": emb[i] for i in range(len(emb))}
        row["class"] = ci
        row.update(priors)
        row["added_utc"] = now_iso
        row["source"] = "manual"
        for k in header:
            row.setdefault(k, "")
        rows.append(row)
        added += 1
    _write_refs(rows, header)
    return jsonify({"ok": True, "added": added, "total": len(rows), "from_record": dlu})


@app.route("/api/knn/remove", methods=["POST"])
def api_knn_remove():
    body = request.get_json(force=True)
    if not app.config["REFS_CSV"]:
        return jsonify({"error": "no refs CSV configured"}), 400
    rows, header = _read_refs()
    if not rows:
        return jsonify({"error": "no references to remove"}), 400
    labels = list(C.ALL_LABELS)
    want = {labels.index(c) for c in body.get("classes", []) if c in labels}
    n = int(body.get("n", 1))
    removed = 0
    for i in range(len(rows) - 1, -1, -1):
        if removed >= n:
            break
        row = rows[i]
        if row.get("source") != "manual":     
            continue
        try:
            ci = int(float(row.get("class", -1)))
        except (TypeError, ValueError):
            continue
        if want and ci not in want:
            continue
        rows.pop(i)
        removed += 1
    _write_refs(rows, header)
    return jsonify({"ok": True, "removed": removed, "total": len(rows)})


def _shell(active, body, page_js, head_extra=""):
    return Response(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Severe-Wx Console</title>{head_extra}
<style>{load_static_file('css/shared.css')}</style></head><body>
<div class="wrap">
  <header>
    <div class="brand"><b>●</b> Severe-Wx Console</div>{load_static_file('html/nav.html')}
    <div class="spacer"></div><select id="loc"></select>
  </header>
  {body}
</div>
<script>{load_static_file('js/shared.js')}
document.querySelectorAll('nav a').forEach(a=>{{ if(a.dataset.p==="{active}") a.classList.add('on'); }});
{page_js}
</script></body></html>""", mimetype="text/html")


@app.route("/")
def page_now():
    return _shell("now", load_static_file('html/now.html'), load_static_file('js/now.js'))


CHART_HEAD = load_static_file('html/chart_head.html')


@app.route("/performance")
def page_performance():
    return _shell("performance", load_static_file("html/performance.html"), load_static_file("js/performance.js"), head_extra=CHART_HEAD)


@app.route("/inputs")
def page_inputs():
    return _shell("inputs",
                  load_static_file("html/inputs_body.html"),
                  load_static_file("js/inputs.js"),
                  head_extra=CHART_HEAD)


@app.route("/admin")
def page_admin():
    return _shell("admin", load_static_file('html/admin.html'), load_static_file('ja/admin.js'))


def main():
    ap = argparse.ArgumentParser(description="Local severe-weather console.")
    ap.add_argument("--logs-dir", default=".", help="dir containing live_log*.json")
    ap.add_argument("--logs", nargs="*", default=None, help="explicit log paths")
    ap.add_argument("--refs-csv", default=None, help="KNN reference CSV (enables admin add/remove)")
    ap.add_argument("--thresholds", default="thresholds.json", help="per-class threshold store")
    ap.add_argument("--emb-dir", default=".", help="dir holding embeddings_<station>.jsonl files")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    paths = args.logs or sorted(glob.glob(os.path.join(args.logs_dir, "live_log*.json")))
    if not paths:
        print(f"[ui] no logs found (live_log*.json in {args.logs_dir}); starting anyway.")
    app.config["LOG_PATHS"] = {_location_key(p): p for p in paths}
    app.config["REFS_CSV"] = args.refs_csv
    app.config["THRESH_PATH"] = args.thresholds
    
    emb = {}
    for k in app.config["LOG_PATHS"]:
        cand = os.path.join(args.emb_dir, f"embeddings_{k}.jsonl")
        emb[k] = cand
    app.config["EMB_PATHS"] = emb
    for k, p in app.config["LOG_PATHS"].items():
        print(f"[ui] location '{k}' <- {p}  (emb: {emb[k]})")
    if args.refs_csv:
        print(f"[ui] refs csv: {args.refs_csv} (admin add/remove enabled)")
    print(f"[ui] thresholds: {args.thresholds}")
    print(f"[ui] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
