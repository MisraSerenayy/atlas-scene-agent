from flask import Flask, request, jsonify
from pathlib import Path
import json, os, sys, traceback, time
from ai_agent import run_prompt


WATCH_PATH = r"C:\GITCLONE\atlas-scene-agent\live_scene.json"

APP_PORT = 5544

app = Flask(__name__)

# ---------- utils ----------
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _safe_float(v, default=0.0):
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return f
    except Exception:
        return default

def _parse_points(pts):
    if isinstance(pts, str):
        out = []
        for tok in pts.strip().split():
            a, b = tok.split(",") if "," in tok else (None, None)
            try:
                out.append([float(a), float(b)])
            except Exception:
                pass
        return out
    if isinstance(pts, list):
        out = []
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                try:
                    out.append([float(p[0]), float(p[1])])
                except Exception:
                    pass
        return out
    return []

def _bbox(points):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def _object_to_rect(o):
    """Return a dict with x,y,w,h for any reasonable shape."""
    t = str(o.get("type") or o.get("shape") or "").lower()
    # Already rect-like center form
    # 1. Rect with top-left (for SVG <rect>)
    if t == "rect" and all(k in o for k in ("x","y","w","h")):
        x = _safe_float(o["x"]); y = _safe_float(o["y"])
        w = _safe_float(o["w"], 1.0); h = _safe_float(o["h"], 1.0)
        return {"x": x + w/2.0, "y": y + h/2.0, "w": max(w,0.001), "h": max(h,0.001)}

    # 2. Already rect-like center form (for your agent/editor)
    if all(k in o for k in ("x","y","w","h")):
        cx = _safe_float(o["x"]); cy = _safe_float(o["y"])
        w  = _safe_float(o["w"], 1.0); h = _safe_float(o["h"], 1.0)
        return {"x": cx, "y": cy, "w": max(w,0.001), "h": max(h,0.001)}
    # Circle
    if t == "circle" and all(k in o for k in ("cx","cy","r")):
        cx = _safe_float(o["cx"]); cy = _safe_float(o["cy"]); r = _safe_float(o["r"], 0.5)
        d = max(2*r, 0.001)
        return {"x": cx, "y": cy, "w": d, "h": d}
    # Ellipse
    if t == "ellipse" and all(k in o for k in ("cx","cy","rx","ry")):
        cx = _safe_float(o["cx"]); cy = _safe_float(o["cy"])
        rx = _safe_float(o["rx"], 0.5); ry = _safe_float(o["ry"], 0.5)
        return {"x": cx, "y": cy, "w": max(2*rx,0.001), "h": max(2*ry,0.001)}
    # Polygon / polyline
    if t in ("polygon","polyline") and o.get("points"):
        pts = _parse_points(o["points"])
        if len(pts) >= 2:
            minx, miny, maxx, maxy = _bbox(pts)
            w = max(maxx - minx, 0.001); h = max(maxy - miny, 0.001)
            return {"x": minx + w/2.0, "y": miny + h/2.0, "w": w, "h": h}
    # Fallback from any plausible fields
    # Accept center form if present
    if "x" in o and "y" in o:
        cx = _safe_float(o.get("x", 0.0)); cy = _safe_float(o.get("y", 0.0))
        w = _safe_float(o.get("w", 1.0), 1.0); h = _safe_float(o.get("h", 1.0), 1.0)
        return {"x": cx, "y": cy, "w": max(w,0.001), "h": max(h,0.001)}
    # Nothing usable
    return {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

def normalize_for_watcher(model: dict):
    """Normalize model to the Blender watcher schema."""
    cv = model.get("canvas") or {}
    grid_w = _safe_float(cv.get("width_m", model.get("grid_w", 40.0)), 40.0)
    grid_h = _safe_float(cv.get("height_m", model.get("grid_h", 30.0)), 30.0)

    objs = model.get("objects") or {}
    payload_extra = {}
    if isinstance(model, dict) and "_source" in model:
        payload_extra["_source"] = str(model["_source"])
    out = {}

    if isinstance(objs, dict):
        items = objs.items()
    elif isinstance(objs, list):
        # convert list to label->object dict using id/label or OBJ_i
        items = []
        for i, o in enumerate(objs):
            key = str(o.get("label") or o.get("id") or f"OBJ_{i+1}")
            items.append((key, o))
    else:
        items = []

    for key, a in items:
        if not isinstance(a, dict):
            continue
        rect = _object_to_rect(a)
        item = {
            "primitive": str(a.get("primitive") or "cube"),
            "x": _safe_float(rect["x"]),
            "y": _safe_float(rect["y"]),
            "w": _safe_float(rect["w"], 1.0),
            "h": _safe_float(rect["h"], 1.0),
            "height": _safe_float(a.get("height", 0.5), 0.5),
        }
        if "z_offset" in a:
            item["z_offset"] = _safe_float(a["z_offset"])
        if "rot_deg" in a:
            item["rot_deg"] = a["rot_deg"]
        out[str(key)] = item

    return {"grid_w": grid_w, "grid_h": grid_h, "objects": out, **payload_extra}

def write_spec_to_watch(spec: dict) -> int:
    payload = normalize_for_watcher(spec)
    payload["_t"] = time.time()  # force content change for watcher
    ensure_dir(WATCH_PATH)
    with open(WATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return len(payload.get("objects", {}))

def standardize_for_agent(model: dict) -> dict:
    """Convert any shapes to x/y/w/h so the agent never KeyErrors on 'x'."""
    if not isinstance(model, dict):
        return {}
    spec = json.loads(json.dumps(model))  # deep copy
    objs = spec.get("objects") or []
    arr = []
    if isinstance(objs, dict):
        it = [{"label": k, **(v if isinstance(v, dict) else {})} for k, v in objs.items()]
    else:
        it = list(objs) if isinstance(objs, list) else []
    for i, o in enumerate(it):
        r = _object_to_rect(o)
        arr.append({
            "label": o.get("label") or o.get("id") or f"OBJ_{i+1}",
            "primitive": o.get("primitive", "cube"),
            "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"],
            "height": _safe_float(o.get("height", 0.5), 0.5),
            "z_offset": _safe_float(o.get("z_offset", 0.0), 0.0),
            "rot_deg": o.get("rot_deg")
        })
    spec["objects"] = arr
    return spec

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp

# ---------- routes ----------
@app.route("/publish", methods=["POST", "OPTIONS"])
def publish():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        model = request.get_json(force=True, silent=False) or {}
        count = write_spec_to_watch(model)
        return jsonify({"ok": True, "written": WATCH_PATH, "objects": count})
    except Exception as e:
        print("PUBLISH ERROR:", e)
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

@app.route("/agent", methods=["POST", "OPTIONS"])
def agent():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        data = request.get_json(force=True, silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"ok": False, "error": "Missing 'prompt'"}), 400

        base_model = data.get("model")
        base_model_std = standardize_for_agent(base_model or {})

        outs = run_prompt(prompt, base_model=base_model_std)

        json_path = outs.get("json")
        if not json_path:
            return jsonify({"ok": False, "error": "Agent did not provide 'json' path"}), 500

        jp = Path(json_path)
        if not jp.exists():
            alt = Path(__file__).resolve().parent / jp.name
            if alt.exists():
                jp = alt
            else:
                return jsonify({"ok": False, "error": f"JSON not found: {jp}"}), 500

        with open(jp, "r", encoding="utf-8") as f:
            spec = json.load(f)
        # tag source so Blender log shows src=agent
        if isinstance(spec, dict):
            spec["_source"] = "agent"

        obj_count = write_spec_to_watch(spec)

        svg_text = None
        svg_path = outs.get("svg")
        if svg_path and os.path.exists(svg_path):
            with open(svg_path, "r", encoding="utf-8") as fh:
                svg_text = fh.read()

        return jsonify({
            "ok": True,
            "written": WATCH_PATH,
            "objects": obj_count,
            "svg": svg_path,
            "svg_text": svg_text,
            "json": str(jp),
            "spec": spec
        })
    except Exception as e:
        print("AGENT ERROR:", e)
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=APP_PORT, debug=True)
