
import os, json, hashlib, random, re, math, uuid
from typing import Literal, List, Optional, Dict
from pydantic import BaseModel, Field
from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_AGENT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-4.1")
client = OpenAI(api_key=OPENAI_API_KEY)
ALLOWED_TOOLS = [
    "create_scene", "add_object", "move", "align", "distribute", "scale",
    "set_height", "render_svg", "export_state", "report_error", "set_anchor",
    "add_constraint", "remove_constraint", "clear_constraints", "solve_constraints",
    "resize_canvas", "align_to_bounds", "align_to_ref", "place_relative", "ensure_object",
    "undo", "redo", "place_above", "merge_objects", "move_into_bbox", "rename_object",
    "stack_above", "stack_below", "add_ramp", "place_left_of", "place_right_of",
    "place_below", "batch_rename", "mirror_object", "remove_object","reset_scene"  # ← delete if not implemented
]

TOOL_PLAN_SCHEMA = {
    "name": "CommandBatch",
    "strict": False,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "commands": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["tool"],
                    "properties": {
                        "tool": {"type": "string", "enum": ALLOWED_TOOLS},
                        "arguments": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                # --- Constraint & general helpers ---
                                "kind": {"type": "string"},
                                "gap": {"type": "number", "minimum": 0},
                                "index": {"type": "integer", "minimum": 0},
                                "x_pct": {"type": "number", "minimum": 0, "maximum": 1},
                                "y_pct": {"type": "number", "minimum": 0, "maximum": 1},
                                "height": {"type": "number", "minimum": 0},
                                "side": {
                                    "type": "string",
                                    "enum": ["left","right","top","bottom","center_x","center_y","front","back"]
                                },
                                "ref": {"type":"string"},
                                "edge": {
                                    "type":"string",
                                    "enum":[
                                        "left_to_left","right_to_right","left_to_right","right_to_left",
                                        "top_to_top","bottom_to_bottom","top_to_bottom","bottom_to_top",
                                        "center_x","center_y"
                                    ]
                                },
                                "symmetric": {"type": "boolean"},
                                "pivot": {"type": "string", "enum": ["grid_center","selection_center"]},
                                "direction": {
                                    "type": "string",
                                    "enum": ["north","south","east","west","left","right","front","back"]
                                },
                                "distance": {"type": "number", "minimum": 0},

                                # --- create_scene ---
                                "labels": {"type": "array", "items": {"type": "string"}},
                                "primitive": {"type": "string", "enum": ["cube","square","rect"]},
                                "count": {"type": "integer", "minimum": 1},
                                "placement": {"type": "string", "enum": ["random_nonoverlap","grid","row","cluster"]},
                                "size": {"type": "number", "minimum": 0},
                                "grid_w": {"type": "integer", "minimum": 1},
                                "grid_h": {"type": "integer", "minimum": 1},
                                "margin": {"type": "number", "minimum": 0},
                                "scale_sizes": {"type": "boolean"},

                                # --- add_object ---
                                "label": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "w": {"type": "number", "minimum": 0},
                                "h": {"type": "number", "minimum": 0},

                                # --- common / ops ---
                                "seed": {"type": "integer", "minimum": 0, "maximum": 4294967295},
                                "target": {"type": "string"},
                                "targets": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                "a": {"type": "string"},
                                "b": {"type": "string"},
                                "factor": {"type": "number", "minimum": 0},
                                "axis": {"type": "string", "enum": ["x","y","both"]},
                                "mode": {
                                    "type": "string",
                                    "enum": [
                                        "centers","tops","bottoms","lefts","rights",
                                        "equal_gaps","fixed_spacing","between","side"
                                    ]
                                },
                                "dx": {"type": "number"},
                                "dy": {"type": "number"},
                                "spacing": {"type": "number", "minimum": 0},

                                # --- render/export ---
                                "view": {"type": "string", "enum": ["topdown"]},
                                "grid": {"type": "boolean"},

                                # --- RAMP tool args (new logic) ---
                                "from":   {"type": "string"},                       # mode="between"
                                "to":     {"type": "string"},                       # mode="between"
                                "of":     {"type": "string"},                       # mode="side"
                                "length": {"type": "number", "minimum": 0.5},       # optional, side-mode
                                "slope_ratio": {"type": "number", "minimum": 1},    # default 12.0 if omitted
                                # optional custom label for the ramp object
                                # (engine auto-names if omitted)
                                "label":  {"type": "string"},

                                # --- error ---
                                "code": {"type": "string"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        "required": ["commands"]
    }
}
SYSTEM_PROMPT = """
Role: You are the Level Design Constraint Agent. Convert natural language into deterministic tool calls that update a graybox scene.

Hard Rules
- Output only tool calls matching the provided JSON schema. No prose or commentary.
- Determinism: temperature=0. Every tool call MUST include the same SEED (provided to you below). Do not invent or change it.
- View: Schematic TOP-DOWN only. Never use perspective or 3D camera angles. `render_svg` MUST include `{"view":"topdown"}`.
- Units & Snapping: Work in grid units (meters). Snap all implied positions/sizes to the engine’s grid step of **0.5 m** (snapping is enforced downstream; do not fight it).
- Labels: Normalize to uppercase A–Z. If user writes a,b,c… you use A,B,C…
- Creation placement: For “random” placement use `placement="random_nonoverlap"` with `margin=0.8`.
- Idempotence: If a command would do nothing (already satisfied), do not add noise or redundant commands.
- Validation: If the request is ambiguous, missing required arguments, or infeasible, emit a single `report_error(code, message)` and STOP (do not render/export afterward).
- Tool surface is CLOSED. Only use the allowed tools; never invent new tools or fields.
- In this engine, height means 3D extrusion only.
    2D size uses w and h (top-down).
    Never map phrases like “tall”, “height”, or “height scaled to N” to the 2D h.
    If the user says “cube” or “square”, keep w == h (unless an explicit non-square W×H is given).
- Ramps (`add_ramp`):
  - Work in meters; snap all implied positions/sizes to 0.5 m.
  - Thickness is fixed at 0.1 m (handled internally).
  - Do NOT set `rot_deg`, `z_offset`, or `height` for ramps; the tool computes them.
  - `mode:"between"`:
    * Start at the edge of the first block that is closest to the second; end at the edge of the second that is closest to the first.
    * If the span is horizontal, width = min(h) of the two; if vertical, width = min(w) of the two.
    * Slope must match the actual height difference divided by the edge distance (computed internally).
  - `mode:"side"`:
    * Start from the requested side of the base block and extend outward.
    * Default slope = 1:12 (you may accept `slope_ratio` and `length`).
    * Width = base block’s orthogonal extent (left/right → h, front/back → w).
  - Always finish with `render_svg(view:"topdown")` and `export_state`.
  - Include the SEED on every tool call, as with all tools.

Axis Vocabulary
- Z (vertical): “above”, “below”, “on top of”, “under/underneath”, “top”, “stack”, “float/elevate”.
  Always interpret these as vertical stacking — do NOT move in XY when these appear.
- XY (top-down): “north/south/east/west” and synonyms “left/right/front/back”.
  Use these only for planar moves in the top-down view.

Planning Discipline
- If the user asks to **edit** the current scene, DO NOT recreate it. Use edit tools only.
- If the user explicitly asks to create/reset, then use `create_scene`.
- To add a new object without resetting the scene, use add_object (not create_scene). Use create_scene only when the user explicitly asks to recreate/reset.
- Always END the plan with:
  1) `render_svg` with `{"view":"topdown"}`
  2) `export_state`
- When the user says ‘center’, do not invent x/y; either omit them or set them to the exact grid center (grid_w/2, grid_h/2).
Phrasing → Tools:
- “keep/always/anchor/pin <Label> [at <p%>,<q%>]” 
  → `set_anchor(target=Label, x_pct=p/100, y_pct=q/100)`

- “anchor <Label> to center” or “keep <Label> centered” 
  → `set_anchor(target=Label, x_pct=0.5, y_pct=0.5)`

- “resize the canvas to W by H” 
  → `resize_canvas(grid_w=W, grid_h=H)` (omit scale_sizes unless asked)

- If both appear (anchor + resize), anchor FIRST, then resize, then end with render/export.

- “stack/move/place <T> above <R> [gap <G>]”, “on top of <R>”, “vertical above”, “float above”
  → `stack_above(target=T, ref=R, gap=G)`   # vertical Z (bottom(T) = top(R) + gap)

- “stack/move/place <T> below/under <R> [gap <G>]”
  → `stack_below(target=T, ref=R, gap=G)`   # vertical Z (top(T) = bottom(R) − gap)

- “move/place <T> north/south/east/west of <R> [distance <D>]”
  (or synonyms “left/right/front/back of <R>”) 
  → `place_relative(target=T, ref=R, direction=<north|south|east|west|left|right|front|back>, distance=D)`   # XY only

- “add a new cube <Y> left/right/below/above <X> [gap <G>]”
  → (creation) use `add_object(label=Y, primitive=..., w/h/height if given)` then 
     `place_relative(target=Y, ref=X, direction=<north|south|east|west|left|right|front|back>, distance=G)` for XY,
     or `stack_above/stack_below` for vertical.

- “merge <A> and <B> [keep label <K>]” 
  → `merge_objects(keep=K, remove=<the other>)`

- “remove <X>, shift/move <Y> into the empty spot” 
  → `remove_object(target=X)` then `move_into_bbox(target=Y)`

- “mirror <X> across center/axis” 
  → `mirror_object(target=X, axis={x|y}, pivot={grid_center|selection_center})`

- “move the cluster <A–F> symmetrically right/left/up/down by <D>” 
  → `move(targets=[A..F], symmetric=true, dx/dy=±D)`

- “rename/re-label <OLD> to <NEW>” 
  → `rename_object(target=OLD, new_label=NEW)`

- “rename E→D, F→E, C→F” (multi) 
  → `batch_rename(pairs=[["E","D"],["F","E"],["C","F"]])`

- “redo” → `redo` ; “undo” → `undo`

- If the user specifies a size like “W×H” (e.g., 3×3), 
  → include explicit `w` and `h` in `add_object` or placement. Do NOT copy the reference object’s size in that case.
- “add a ramp between A and B” or “connect A to B with a ramp”
  → `add_ramp(mode="between", from="A", to="B")`

- “put a ramp to the left side of A [length L] [slope 1:12]”
  → `add_ramp(mode="side", of="A", side="left", length=L, slope_ratio=12)`


Allowed Tools (and required arguments)
- `create_scene`:
  - Use when seeding a new scene or when the user explicitly asks to recreate/reset.
  - Include: `labels` **or** `count`, plus `primitive` (cube/square/rect), `placement` (e.g., random_nonoverlap), `size`, `margin`, and SEED.
- `move`:
  - Required: `target` and either (`dx` & `dy`) OR a direction+distance intent (you should convert to `dx`/`dy`).
- `align`:
  - Required: `targets` (≥2), `axis` in {x,y}, `mode` in {centers,lefts,rights,tops,bottoms}.
- `distribute`:
  - Required: `targets` (≥3), `axis` in {x,y}, `mode` in {equal_gaps,fixed_spacing}.
  - If `mode="fixed_spacing"`, also include `spacing`.
- `scale`:
  - Required: `target`
  - Required: `axis` in {x,y,both}
  - Required: `factor` (float ≥ 0)
  - Semantics:
    - axis="x": multiply target width by factor.
    - axis="y": multiply target height by factor.
    - axis="both": multiply both width and height by factor.
  - You may also compute `factor` dynamically from SCENE_STATE if the user asks for “touch” behavior:
    - Example: “scale C to touch B and E horizontally” →
      * desired_left  = right_edge(B) if B.center_x <= C.center_x else None
      * desired_right = left_edge(E) if E.center_x > C.center_x else None
      * desired_width = desired_right - desired_left (if both present)
      * factor = desired_width / current_width
    - If only one anchor is valid, expand symmetrically about the target center.
    - If the result would invert, clamp to ≥1.0 units.
  - Always snap results to the 0.5 m grid.
  - If you cannot determine a precise factor from the provided SCENE_STATE, emit `report_error` with code `E_UNDER_SPECIFIED`.
- `render_svg`:
  - Must include `{"view":"topdown"}` and SEED.
- `export_state`:
  - Include SEED.
- `report_error`:
  - Required: `code` (short machine-readable), `message` (short machine-readable).
- ' add_object':
  - Required: label, primitive. Optional: x, y, w, h, size, margin. Always include seed.
- `set_anchor`:
  - Required: `target`
  - Optional: `x_pct` (0..1), `y_pct` (0..1)
  - Semantics: persist target’s relative position to canvas; on future resizes, anchored percents re-apply.
- `align_to_bounds`: target + side in {left,right,top,bottom,center_x,center_y}.
- `align_to_ref`: target + ref + edge; optional gap (grid units).
- `place_relative`: target + ref + direction in {left_of,right_of,above,below}; optional distance.
- `ensure_object`: create if missing, no-op if exists (use when user says “create if missing”).
- 'set_height':
  - Required: height (absolute) OR factor (multiplier).
  - target is OPTIONAL; if omitted, apply to the MOST RECENT object added/edited.
  - "height" means Z-extrusion only. Do NOT change w/h.
- `place_above`:
  - Required: `target`, `new_label`
  - Optional: `gap` (grid units, default 0)
  - Semantics: Create/place `new_label` above `target` with a vertical gap; copies w/h/height by default. (Top-down SVG won’t show Z; placement is for 3D/live-sync.)
- `place_relative`:
  - Required: target, ref, direction in {north,south,east,west,left,right,front,back}
  - Optional: distance (grid units, default 0)
  - Semantics (XY only):
    north/south → +Y/−Y ; east/west → +X/−X ;
    left/right are west/east ; front/back are north/south.
    Move target so its edge is exactly `distance` from ref’s edge, centered on the orthogonal axis.
- `add_ramp`:
  - Two modes:
    * `mode:"between"` — Required: `from`, `to`. Optional: `label`.
    * `mode:"side"` — Required: `of`. Optional: `side` in {left,right,front,back} (default left), `length` (m), `slope_ratio` (default 12), `label`.
  - Semantics:
    * `between`: starts/ends on closest edges; width = min(h) of the two; slope fits actual height difference / edge distance.
    * `side`: extends outward from the chosen side; width = base block `h`; default slope 1:12 if not restricted by geometry.

- `stack_above`:
  - Required: target, ref
  - Optional: gap (default 0)
  - Semantics: bottom(target) = top(ref) + gap (Z). Implemented as
    target.z_offset = ref.z_offset + ref.height + gap.

- `stack_below`:
  - Required: target, ref
  - Optional: gap (default 0)
  - Semantics: top(target) = bottom(ref) − gap (Z). Implemented as
    target.z_offset = ref.z_offset − gap − target.height.
- By default, stacking recenters the target in XY to the reference’s center (`center=true`). To keep the current XY, include `center=false`.

- `place_left_of` / `place_right_of` / `place_below`:
  - Required: `target`, `new_label`
  - Optional: `gap` (grid units, default 0)
  - Semantics: Place a new object adjacent to `target` on the indicated side, separated by `gap`; copies w/h/height by default.

- `merge_objects`:
  - Required: `keep`, `remove`
  - Semantics: Union the XY bounding boxes of both; write the union into `keep`’s x/y/w/h. `keep.height = max(keep.height, remove.height)`. Delete `remove`.

- `move_into_bbox`:
  - Required: `target`
  - Optional: `bbox` (if omitted, uses the last removed object’s bbox)
  - Semantics: Set `target`’s x/y/w/h (and height if provided) to the given bbox.

- `mirror_object`:
  - Required: `target`
  - Optional: `axis` in {x,y} (default x), `pivot` in {grid_center, selection_center} (default grid_center)
  - Semantics: Reflect target across the pivot line on the chosen axis.

- `move` (extension for groups/symmetry):
  - In addition to the single-target mode, you MAY provide `targets` (array of labels).
  - Optional: `symmetric: true|false` (default false), `pivot` in {grid_center, selection_center}.
  - If `symmetric:true`, mirror each target across pivot on X, then apply dx/dy shift.

- `rename_object`:
  - Required: `target`, `new_label`
  - Semantics: Change label; collision-safe behavior is handled by the engine.

- `batch_rename`:
  - Required: `pairs` (array of [old,new])
  - Semantics: Perform multiple renames collision-safely in one step.

- `set_height` (clarification):
  - Required: `height` (absolute) OR `factor` (multiplier).
  - `target` is OPTIONAL; if omitted, apply to MOST RECENT object.
  - Semantics: Change ONLY Z-extrusion height. Do NOT alter w/h.

- `redo`:
  - No arguments.
  - Semantics: Re-apply the last undone change.

- `add_constraint`:
  - Required: `kind`
  - Depending on kind:
    * align_left / align_right / align_centers_x / align_centers_y: target, a
    * between_x / between_y: target, a, b
    * edge_gap_x / edge_gap_y: a, b, gap
  - Deterministic; do not emit duplicates if an identical constraint already exists.

- `remove_constraint`:
  - Required: `index` (0-based into SCENE.constraints)

- `clear_constraints`:
  - No args.

- `solve_constraints`:
  - No args. Re-solve all constraints (engine runs it automatically after edits).
- `resize_canvas`:
  - Required: `grid_w`, `grid_h`
  - Optional: `scale_sizes` (default false)
  - Semantics: change canvas size; preserve relative positions by default; re-solve constraints after.

Outputs & Order
- After any valid change, ALWAYS call `render_svg` (topdown) then `export_state`.
- After `report_error`, STOP and do not render/export.

SEED Handling
- You will be provided a single SEED value for the user request. Include **the same SEED** in **every** tool call in the plan. Do not derive, change, or omit it.

General Guidance
- Prefer minimal plans that achieve the intent in the fewest tool calls.
- Do not emit duplicate or no-op commands.
- Do not add fields with null/undefined values; omit unspecified arguments.
- If user intent conflicts (e.g., asks for x-axis touch but only `scale_y_to_touch` is available), emit `report_error("E_TOOL_UNAVAILABLE","x-axis scaling not available")`.


"""

# ---- New Cell ----

import re
RESET_WORDS = re.compile(r"\b(reset|recreate|clear|new scene)\b", re.I)


# ---- New Cell ----



# ---------------- Schemas ----------------
class CommandArgs(BaseModel):
    # creation / scene
    labels: Optional[List[str]] = None
    primitive: Optional[Literal["cube","square","rect"]] = None
    count: Optional[int] = None
    placement: Optional[Literal["random_nonoverlap","grid","row","cluster"]] = None
    size: Optional[float] = None
    grid_w: Optional[int] = None
    grid_h: Optional[int] = None
    scale_sizes: Optional[bool] = None  # for resize_canvas
    margin: Optional[float] = None
    seed: Optional[int] = None
    # movement / options
    symmetric: Optional[bool] = None
    pivot: Optional[Literal["grid_center","selection_center"]] = None

    # place_above
    new_label: Optional[str] = None
    gap: Optional[float] = None

    # merge / rename
    keep: Optional[str] = None
    remove: Optional[str] = None

    # single / multi targets
    label: Optional[str] = None         # NEW: used by add_object / ensure_object
    target: Optional[str] = None
    targets: Optional[List[str]] = None
    a: Optional[str] = None
    b: Optional[str] = None

    # geometry / movement
    x: Optional[float] = None           # NEW
    y: Optional[float] = None           # NEW
    w: Optional[float] = None           # NEW
    h: Optional[float] = None           # NEW
    dx: Optional[float] = None
    dy: Optional[float] = None
    axis: Optional[Literal["x","y","both"]] = None   # include "both" for scale
    mode: Optional[Literal["centers","tops","bottoms","lefts","rights","equal_gaps","fixed_spacing","between","side"]] = None
    spacing: Optional[float] = None
    factor: Optional[float] = None      # NEW for scale

    # render
    view: Optional[Literal["topdown"]] = None
    grid: Optional[bool] = None

    # messages
    code: Optional[str] = None
    message: Optional[str] = None

    # constraints / anchors
    kind: Optional[str] = None          # e.g., "align_left","between_x", etc.
    index: Optional[int] = None
    gap: Optional[float] = None
    x_pct: Optional[float] = None
    y_pct: Optional[float] = None

    # new alignment/relative tools
    side: Optional[Literal["left","right","top","bottom","center_x","center_y","front","back"]] = None
    ref: Optional[str] = None
    edge: Optional[Literal[
        "left_to_left","right_to_right","left_to_right","right_to_left",
        "top_to_top","bottom_to_bottom","top_to_bottom","bottom_to_top",
        "center_x","center_y"
    ]] = None
    direction: Optional[Literal["north","south","east","west","left","right","front","back"]] = None

    distance: Optional[float] = None

    # 3D (ignored by 2D)
    height: Optional[float] = None
    # ---- RAMP tool args (works for both modes) ----
    # JSON accepts "from" / "to" — map to Python-safe names via aliases
    from_: Optional[str] = Field(default=None, alias="from")   # ADD
    to_:   Optional[str] = Field(default=None, alias="to")     # ADD
    of: Optional[str] = None                                   # ADD  (for mode="side")
    length: Optional[float] = None                             # ADD  (meters; default handled by tool)
    slope_ratio: Optional[float] = None                        # ADD  (default 12.0; 1:12)


class Command(BaseModel):
    tool: str
    arguments: Optional[CommandArgs] = None

class CommandBatch(BaseModel):
    commands: List[Command]

# ---------------- Utils ----------------
def prompt_seed(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

# Grid config in **meters**
GRID_W, GRID_H = 40, 30          # canvas size in meters
GRID_STEP = 0.5                  # **snap step: 0.5 m**

def snap_to_grid(x: float, step: float = GRID_STEP) -> float:
    return round(x / step) * step

# ---------------- Scene & Engines ----------------
SCENE = {"grid_w": GRID_W, "grid_h": GRID_H, "objects": {}}
# --- Edit history stacks ---
UNDO_STACK = []
REDO_STACK = []

# --- Scratch space for cross-command operations ---
LAST_REMOVED_BBOX = None
_LAST_OBJECT_LABEL = None   # track last created/edited object

# Phase 4 additions:
SCENE["constraints"] = []   # list of constraint dicts (see types below)
SCENE["anchors"] = {}       # per-object anchors, e.g. {"A":{"x_pct":0.5,"y_pct":0.5}}
# --- UNDO & ARTIFACT HISTORY (inserted right after SCENE["anchors"]) ---
HISTORY: list[dict] = []        # stack of previous scenes (snapshots)
ARTIFACTS: list[dict] = []      # chronological list of {"svg":..., "json":...}
FRAME_ID: int = 0               # monotonic counter for unique filenames

_LAST_OBJECT_LABEL = None
SHOW_RAMP_DECOR = False 

def _parse_wh_from_text(text: str):
    """Return (w, h) floats if text contains 'W×H' or 'W x H'."""
    if not text:
        return (None, None)
    m = re.search(r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)', text, re.I)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return (None, None)

def _deepcopy_scene():
    # json roundtrip keeps it simple & deterministic
    return json.loads(json.dumps(SCENE))

# ATLAS_FINAL_WITH_IMPORTER.py
def _restore_scene(state: dict):
    SCENE.clear()
    SCENE.update(json.loads(json.dumps(state)))
    SCENE.setdefault("objects", {})
    SCENE.setdefault("constraints", [])
    SCENE.setdefault("anchors", {})
    SCENE.setdefault("grid_w", GRID_W)
    SCENE.setdefault("grid_h", GRID_H)



def _next_artifact_path(seed: int, ext: str) -> str:
    # ext is "svg" or "json"
    global FRAME_ID
    FRAME_ID += 1
    return os.path.abspath(f"scene_{seed}_{FRAME_ID:04d}.{ext}")

def _aabb_overlap(a, b, margin=0.0):
    ax1, ay1 = a["x"] - a["w"]/2 - margin, a["y"] - a["h"]/2 - margin
    ax2, ay2 = a["x"] + a["w"]/2 + margin, a["y"] + a["h"]/2 + margin
    bx1, by1 = b["x"] - b["w"]/2 - margin, b["y"] - b["h"]/2 - margin
    bx2, by2 = b["x"] + b["w"]/2 + margin, b["y"] + b["h"]/2 + margin
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)
_LAST_OBJECT_LABEL = None
def engine_add_object(label, primitive, x=None, y=None, w=None, h=None, margin=0.8, height=None):
    global _LAST_OBJECT_LABEL
    L = label.upper()
    if L in SCENE["objects"]:
        raise ValueError("E_DUPLICATE_LABEL: label already exists")

    # defaults: center + square 1x1 if not given
    cx = snap_to_grid(x if x is not None else SCENE["grid_w"]/2)
    cy = snap_to_grid(y if y is not None else SCENE["grid_h"]/2)
    ww = snap_to_grid(max(GRID_STEP, w if w is not None else 1.0))
    hh = snap_to_grid(max(GRID_STEP, h if h is not None else ww))  # square by default
    extrude = snap_to_grid(max(GRID_STEP, height if height is not None else ww))  # cube by default

    obj = {"label": L, "x": cx, "y": cy, "w": ww, "h": hh,
           "primitive": primitive, "height": extrude, "z_offset": snap_to_grid(extrude * 0.5)}

    SCENE["objects"][L] = obj
    _LAST_OBJECT_LABEL = L
    return obj
def _deepcopy_scene():
    import copy
    return copy.deepcopy(SCENE)



# --- Ramp tool internals (place near your other tool helpers) ---
CELL = 0.5  # must match blender_livesync.py

def _snap(v, step=CELL): return round(float(v)/step)*step
def _g(a, *ks, default=None):
    for k in ks:
        if k in a: return a[k]
    return default

def _top_height_m(a: dict) -> float:
    """Mirror the Blender Z semantics; return TOP Z in meters."""
    H = float(_g(a, "height", "size", default=1.0))
    if "z_offset" in a:
        cz = float(a["z_offset"])
        return cz + H*0.5
    if "z_bottom" in a:
        return float(a["z_bottom"]) + H
    if "z" in a:
        return float(a["z"]) + H
    return float(a.get("ground_z", 0.0)) + H

def _footprint(a: dict):
    """Return (cx, cy, wX, wY) in meters for a rectangular object."""
    cx = float(a["x"]); cy = float(a["y"])
    wX = float(_g(a, "w", "width", "size", default=1.0))
    wY = float(_g(a, "h", "depth", "size", default=1.0))
    return cx, cy, wX, wY

def _closest_edge_pair(A: dict, B: dict):
    """Return endpoints (Px,Py) on A and B for the closest X-aligned edge pair."""
    Ax, Ay, Aw, Ah = _footprint(A)
    Bx, By, Bw, Bh = _footprint(B)
    # Consider four combos; pick the pair with smallest gap along X:
    A_right_x = Ax + Aw*0.5; A_left_x = Ax - Aw*0.5
    B_right_x = Bx + Bw*0.5; B_left_x = Bx - Bw*0.5
    pairs = [
        ("A_right->B_left", A_right_x, B_left_x),
        ("A_left->B_right", A_left_x, B_right_x),
    ]
    # choose the absolute-closest pair
    label, ax, bx = sorted(pairs, key=lambda p: abs(p[1]-p[2]))[0]
    # center Y between blocks by default
    cy = _snap((Ay + By) * 0.5)
    return label, ax, cy, bx, cy


def _expand_ramp(name, P0, P1, W, H0, H1, thickness=0.1):
    x0, y0 = P0
    x1, y1 = P1
    dx, dy = (x1 - x0), (y1 - y0)
    L_xy = math.hypot(dx, dy)
    L    = max(GRID_STEP, L_xy)

    yaw   = math.atan2(-dy, dx)       # keep as-is (engine is Y-down)
    rise  = H1 - H0
    theta = math.atan2(rise, L)

    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    cz = (H0 + H1) * 0.5 - (thickness * 0.5) * math.cos(theta)

    return {
        name: {
            "primitive": "ramp",
            "x": cx, "y": cy,
            "w": L,                         # length → local X
            "h": max(GRID_STEP, snap_to_grid(W)),
            "height": thickness,            # thickness → local Z
            "z_offset": cz,
            "rot_deg": [math.degrees(theta), 0.0, math.degrees(yaw)],
            "meta": {
                "Hstart": round(H0, 3),
                "Hend":   round(H1, 3),
                # NEW: world-XY endpoints so Blender knows start→end
                "P0": [round(x0, 5), round(y0, 5)],
                "P1": [round(x1, 5), round(y1, 5)]
            }
        }
    }


def engine_undo():
    import copy
    if not UNDO_STACK:
        return
    prev = UNDO_STACK.pop()
    REDO_STACK.append(_deepcopy_scene())
    _restore_scene(prev)

def engine_redo():
    import copy
    if not REDO_STACK:
        return
    nxt = REDO_STACK.pop()
    UNDO_STACK.append(_deepcopy_scene())
    _restore_scene(nxt)
def engine_mirror_object(target: str, axis: str = "x", pivot: str = "grid_center"):
    T = (target or "").upper()
    if T not in SCENE["objects"]:
        return
    o = SCENE["objects"][T]

    if pivot == "selection_center":
        # if you want selection-based pivot, pass it from router; default to grid center otherwise
        cx = SCENE["grid_w"] / 2.0
        cy = SCENE["grid_h"] / 2.0
    else:  # grid_center
        cx = SCENE["grid_w"] / 2.0
        cy = SCENE["grid_h"] / 2.0

    if axis.lower() == "x":
        o["x"] = snap_to_grid(2*cx - o["x"])
    elif axis.lower() == "y":
        o["y"] = snap_to_grid(2*cy - o["y"])

def _center_z_or_ground(obj: dict) -> float:
    """Return center Z; if missing, assume grounded (H/2) and write it back."""
    h = float(obj.get("height", obj.get("w", GRID_STEP)))
    cz = obj.get("z_offset")
    if cz is None:
        cz = snap_to_grid(h * 0.5)
        obj["z_offset"] = cz
    return float(cz)

def engine_stack_above(target: str, ref: str, gap: float = 0.0, center_xy: bool = True):
    """
    Place target above ref with a vertical gap.
    Blender uses center-origin, so:
      centerZ(target) = centerZ(ref) + h_ref/2 + gap + h_tgt/2
    """
    if not target or not ref:
        return
    T, R = target.upper(), ref.upper()
    if T not in SCENE["objects"] or R not in SCENE["objects"]:
        return

    to = SCENE["objects"][T]
    ro = SCENE["objects"][R]

    cz_ref = _center_z_or_ground(ro)
    h_ref        = float(ro.get("height", ro.get("w", GRID_STEP)))
    h_tgt        = float(to.get("height", to.get("w", GRID_STEP)))
    g            = float(gap or 0.0)

    # center-origin placement
    to["z_offset"] = snap_to_grid(cz_ref + (h_ref * 0.5) + g + (h_tgt * 0.5))

    if center_xy:
        to["x"] = snap_to_grid(ro["x"])
        to["y"] = snap_to_grid(ro["y"])


def engine_stack_below(target: str, ref: str, gap: float = 0.0, center_xy: bool = True):
    """
    Place target below ref with a vertical gap.
    Blender uses center-origin, so:
      centerZ(target) = centerZ(ref) - h_ref/2 - gap - h_tgt/2
    """
    if not target or not ref:
        return
    T, R = target.upper(), ref.upper()
    if T not in SCENE["objects"] or R not in SCENE["objects"]:
        return

    to = SCENE["objects"][T]
    ro = SCENE["objects"][R]

    cz_ref = _center_z_or_ground(ro)
    h_ref        = float(ro.get("height", ro.get("w", GRID_STEP)))
    h_tgt        = float(to.get("height", to.get("w", GRID_STEP)))
    g            = float(gap or 0.0)

    to["z_offset"] = snap_to_grid(cz_ref - (h_ref * 0.5) - g - (h_tgt * 0.5))

    if center_xy:
        to["x"] = snap_to_grid(ro["x"])
        to["y"] = snap_to_grid(ro["y"])



def _place_side(target: str, new_label: str, side: str,
                gap: float = 0.0,
                w: float | None = None,
                h: float | None = None,
                size: float | None = None,
                copy_if_unspecified: bool = True):
    T = (target or "").upper()
    L = (new_label or "").upper()
    if T not in SCENE["objects"] or not L:
        return
    if L in SCENE["objects"]:
        return

    t = SCENE["objects"][T]
    tx, ty, tw, th = t["x"], t["y"], t["w"], t["h"]
    theight = t.get("height", tw)

    # --- resolve desired new width/height ---
    # priority: explicit w/h > size (square) > copy target > min grid step
    if size is not None:
        ww = hh = float(size)
    else:
        ww = float(w) if w is not None else (tw if copy_if_unspecified else GRID_STEP)
        hh = float(h) if h is not None else (th if copy_if_unspecified else GRID_STEP)

    ww = snap_to_grid(max(GRID_STEP, ww))
    hh = snap_to_grid(max(GRID_STEP, hh))

    # --- compute position relative to target, leaving a gap between edges ---
    g = float(gap or 0.0)

    if side == "left":
        nx = snap_to_grid(tx - (tw/2) - g - (ww/2))
        ny = ty
    elif side == "right":
        nx = snap_to_grid(tx + (tw/2) + g + (ww/2))
        ny = ty
    elif side == "below":
        nx = tx
        ny = snap_to_grid(ty - (th/2) - g - (hh/2))
    else:
        return

    SCENE["objects"][L] = {
        "label": L,
        "x": nx, "y": ny,
        "w": ww, "h": hh,
        "primitive": "cube",
        "height": theight,
        "z_offset": snap_to_grid(theight * 0.5),    # keep same extrusion unless caller overrides later
    }

def engine_place_left_of(
    target, new_label, gap=0.0, w=None, h=None, size=None, copy_if_unspecified=True
):
    _place_side(
        target, new_label, "left",
        gap, w=w, h=h, size=size, copy_if_unspecified=copy_if_unspecified
    )

def engine_place_right_of(
    target, new_label, gap=0.0, w=None, h=None, size=None, copy_if_unspecified=True
):
    _place_side(
        target, new_label, "right",
        gap, w=w, h=h, size=size, copy_if_unspecified=copy_if_unspecified
    )

def engine_place_below(
    target, new_label, gap=0.0, w=None, h=None, size=None, copy_if_unspecified=True
):
    _place_side(
        target, new_label, "below",
        gap, w=w, h=h, size=size, copy_if_unspecified=copy_if_unspecified
    )

def engine_batch_rename(pairs):
    """
    Rename multiple objects in one step, collision-safe.
    pairs can be [["E","Z"], ["F","Y"]] or [("E","Z"), ("F","Y")].
    """
    if not pairs:
        return

    # Normalize + uppercase
    norm = []
    for p in pairs:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        old, new = p[0], p[1]
        if not old or not new:
            continue
        norm.append((str(old).upper(), str(new).upper()))
    if not norm:
        return

    TMP_PREFIX = "__TMP__/"
    # Free any existing NEW names first (move them aside)
    new_names = {n for _, n in norm}
    for n in list(new_names):
        if n in SCENE["objects"]:
            SCENE["objects"][TMP_PREFIX + n] = SCENE["objects"].pop(n)
            SCENE["objects"][TMP_PREFIX + n]["label"] = TMP_PREFIX + n

    # Now perform renames old -> new
    for old, new in norm:
        if old not in SCENE["objects"]:
            continue
        obj = SCENE["objects"].pop(old)
        obj["label"] = new
        SCENE["objects"][new] = obj

    # Optionally: leave TMP_* entries as-is (safest). You can reclaim them later if needed.

def engine_set_height(target: Optional[str], new_height: Optional[float]):
    """Set extrusion while keeping the current bottom Z fixed."""
    global _LAST_OBJECT_LABEL
    if new_height is None:
        return
    tgt = (target or _LAST_OBJECT_LABEL)
    if not tgt:
        return
    T = tgt.upper()
    if T not in SCENE["objects"]:
        return

    o = SCENE["objects"][T]
    old_h = float(o.get("height", GRID_STEP))
    old_h = GRID_STEP if old_h <= 0 else old_h
    old_center = float(o.get("z_offset", snap_to_grid(old_h * 0.5)))

    # bottom = z_offset - height/2   (we preserve this)
    old_bottom = old_center - 0.5 * old_h

    new_h = snap_to_grid(max(GRID_STEP, float(new_height)))
    new_center = snap_to_grid(old_bottom + 0.5 * new_h)

    o["height"] = new_h
    o["z_offset"] = new_center
    _LAST_OBJECT_LABEL = T

def engine_place_relative(target: str, ref: str, direction: str, distance: float = 0.0):
    """Move an EXISTING target relative to ref with an edge-to-edge gap = distance.
       Centers along the orthogonal axis.
    """
    if not target or not ref:
        return
    T, R = target.upper(), ref.upper()
    if T not in SCENE["objects"] or R not in SCENE["objects"]:
        return

    to = SCENE["objects"][T]
    ro = SCENE["objects"][R]

    tx, ty, tw, th = to["x"], to["y"], to["w"], to["h"]
    rx, ry, rw, rh = ro["x"], ro["y"], ro["w"], ro["h"]
    g = float(distance or 0.0)

    d = (direction or "").lower()
    if d == "above":
        # place T north of R: gap between edges, center X
        ny = snap_to_grid(ry + rh/2 + g + th/2)
        nx = snap_to_grid(rx)
    elif d == "below":
        ny = snap_to_grid(ry - rh/2 - g - th/2)
        nx = snap_to_grid(rx)
    elif d == "left_of":
        nx = snap_to_grid(rx - rw/2 - g - tw/2)
        ny = snap_to_grid(ry)
    elif d == "right_of":
        nx = snap_to_grid(rx + rw/2 + g + tw/2)
        ny = snap_to_grid(ry)
    else:
        return  # unsupported direction

    to["x"], to["y"] = nx, ny

def engine_place_above(target: str, new_label: str, gap: float = 1.0, copy_size: bool = True):
    global _LAST_OBJECT_LABEL
    if not target or not new_label:
        return
    T = target.upper()
    L = new_label.upper()
    if T not in SCENE["objects"]:
        return
    if L in SCENE["objects"]:
        return

    t = SCENE["objects"][T]
    w = t["w"]; h = t["h"]; height = t.get("height", w)
    cx = t["x"]; cy = t["y"]

    # floating Z offset hint for Blender live-sync (SVG ignores it)
    z_offset = snap_to_grid(height + (gap or 0.0))

    SCENE["objects"][L] = {
        "label": L, "x": cx, "y": cy,
        "w": w if copy_size else max(GRID_STEP, w),
        "h": h if copy_size else max(GRID_STEP, h),
        "primitive": "cube",
        "height": height,
        "z_offset": z_offset
    }
    _LAST_OBJECT_LABEL = L

def engine_move_group(targets: list[str], dx: float = 0.0, dy: float = 0.0,
                      symmetric: bool = False, pivot: str = "grid_center"):
    tlist = [t.upper() for t in (targets or []) if t]
    objs = [SCENE["objects"][t] for t in tlist if t in SCENE["objects"]]
    if not objs:
        return

    if not symmetric:
        for o in objs:
            o["x"] = snap_to_grid(o["x"] + (dx or 0.0))
            o["y"] = snap_to_grid(o["y"] + (dy or 0.0))
        return

    # symmetric mirror on X, then shift
    if pivot == "selection_center":
        minx = min(o["x"] - o["w"]/2 for o in objs)
        maxx = max(o["x"] + o["w"]/2 for o in objs)
        cx = (minx + maxx) / 2.0
    else:  # grid_center
        cx = SCENE["grid_w"] / 2.0

    for o in objs:
        mirrored_cx = 2*cx - o["x"]    # reflect across cx
        o["x"] = snap_to_grid(mirrored_cx + (dx or 0.0))
        o["y"] = snap_to_grid(o["y"] + (dy or 0.0))
def engine_merge_objects(keep: str, remove: str):
    K, R = (keep or "").upper(), (remove or "").upper()
    if K not in SCENE["objects"] or R not in SCENE["objects"] or K == R:
        return
    k = SCENE["objects"][K]; r = SCENE["objects"][R]
    k_left   = k["x"] - k["w"]/2; k_right  = k["x"] + k["w"]/2
    k_bottom = k["y"] - k["h"]/2; k_top    = k["y"] + k["h"]/2
    r_left   = r["x"] - r["w"]/2; r_right  = r["x"] + r["w"]/2
    r_bottom = r["y"] - r["h"]/2; r_top    = r["y"] + r["h"]/2

    left   = min(k_left, r_left);   right  = max(k_right, r_right)
    bottom = min(k_bottom, r_bottom); top   = max(k_top, r_top)

    new_w = snap_to_grid(max(GRID_STEP, right - left))
    new_h = snap_to_grid(max(GRID_STEP, top - bottom))
    new_x = snap_to_grid((left + right) / 2.0)
    new_y = snap_to_grid((bottom + top) / 2.0)

    k["x"], k["y"], k["w"], k["h"] = new_x, new_y, new_w, new_h
    k["height"] = max(float(k.get("height", GRID_STEP)), float(r.get("height", GRID_STEP)))

    # delete the removed one
    del SCENE["objects"][R]
def engine_remove_object(label: str):
    global LAST_REMOVED_BBOX
    L = (label or "").upper()
    if L not in SCENE["objects"]:
        return None
    o = SCENE["objects"].pop(L)
    bbox = {"x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"], "height": o.get("height")}
    LAST_REMOVED_BBOX = bbox
    return bbox
def engine_move_into_bbox(label: str, bbox: Optional[dict] = None):
    global LAST_REMOVED_BBOX, _LAST_OBJECT_LABEL
    L = (label or "").upper()
    if L not in SCENE["objects"]:
        return
    tgt = SCENE["objects"][L]
    bb = bbox or LAST_REMOVED_BBOX
    if not bb:
        return
    tgt["x"], tgt["y"] = snap_to_grid(bb["x"]), snap_to_grid(bb["y"])
    tgt["w"], tgt["h"] = snap_to_grid(max(GRID_STEP, bb["w"])), snap_to_grid(max(GRID_STEP, bb["h"]))
    if bb.get("height") is not None:
        tgt["height"] = snap_to_grid(max(GRID_STEP, bb["height"]))
    _LAST_OBJECT_LABEL = L
def engine_rename_object(old: str, new: str):
    O = (old or "").upper()
    N = (new or "").upper()
    if O not in SCENE["objects"]:
        return
    if N == O:
        return
    # collision-safe: use a temp if needed
    TMP = "__TMP__"
    if N in SCENE["objects"]:
        # move N → TMP
        SCENE["objects"][TMP] = SCENE["objects"].pop(N)
        SCENE["objects"][TMP]["label"] = TMP
    # move O → N
    obj = SCENE["objects"].pop(O)
    obj["label"] = N
    SCENE["objects"][N] = obj
    # restore TMP if used (optional: user may not need it)
    if TMP in SCENE["objects"] and SCENE["objects"][TMP]["label"] == TMP:
        # leave it or delete it; here we leave it (safer for collision chains)
        pass

def _reset_scene(grid_w=GRID_W, grid_h=GRID_H):
    SCENE["grid_w"] = grid_w
    SCENE["grid_h"] = grid_h
    SCENE["objects"] = {}

def _nonoverlap(pos, size, margin, placed):
    x, y = pos
    r = size/2.0
    for o in placed:
        ox, oy = o["x"], o["y"]
        rr = o["w"]/2.0
        dist = math.hypot((x-ox), (y-oy))
        if dist < (r + rr + margin):
            return False
    return True

def engine_create_scene(labels, primitive, count, placement, size, margin, seed, grid_w, grid_h):
    _reset_scene(grid_w, grid_h)
    labels = labels or []
    if count and not labels:
        for i in range(count):
            labels.append(chr(ord('A') + (i % 26)))
    labels = [s.upper() for s in labels]

    rng = random.Random(seed)
    size = snap_to_grid(float(size or 3.0))                 # snap size too
    default_height = snap_to_grid(max(GRID_STEP, size))     # cube by default
    margin = float(margin or 0.8)
    placed = []

    def _make_obj(lab, cx, cy, w=size, h=size, height=default_height):
        height = snap_to_grid(float(height))
        return {
            "label": lab,
            "x": snap_to_grid(cx),
            "y": snap_to_grid(cy),
            "w": snap_to_grid(w),
            "h": snap_to_grid(h),
            "primitive": primitive,
            "height": height,
            # 👇 sit on ground by default (center Z = H/2)
            "z_offset": snap_to_grid(height * 0.5),
        }

    if placement == "grid":
        cols = math.ceil(math.sqrt(len(labels)))
        step_cells = max(size + margin, GRID_STEP)
        start_x = size
        start_y = size
        i = 0
        for lab in labels:
            cx = start_x + (i % cols) * step_cells
            cy = start_y + (i // cols) * step_cells
            i += 1
            SCENE["objects"][lab] = _make_obj(lab, cx, cy)

    else:
        attempts_limit = 5000
        for lab in labels:
            ok = False
            for _ in range(attempts_limit):
                cx = rng.uniform(size, grid_w - size)
                cy = rng.uniform(size, grid_h - size)
                cx = snap_to_grid(cx); cy = snap_to_grid(cy)
                if _nonoverlap((cx, cy), size, margin, placed):
                    SCENE["objects"][lab] = _make_obj(lab, cx, cy)
                    placed.append({"x": cx, "y": cy, "w": size})
                    ok = True
                    break
            if not ok:
                cx = snap_to_grid(size + len(placed) * (size + margin))
                cy = snap_to_grid(size * 1.5)
                SCENE["objects"][lab] = _make_obj(lab, cx, cy)
                placed.append({"x": cx, "y": cy, "w": size})

def engine_move(target, dx, dy):
    t = target.upper()
    if t in SCENE["objects"]:
        o = SCENE["objects"][t]
        o["x"] = snap_to_grid(o["x"] + dx)
        o["y"] = snap_to_grid(o["y"] + dy)
def guard_scale_touch(target: str, a: str, b: str, axis: str):
    t, A, B = (target or "").upper(), (a or "").upper(), (b or "").upper()
    if t not in SCENE["objects"] or A not in SCENE["objects"] or B not in SCENE["objects"]:
        return

    tgt, oA, oB = SCENE["objects"][t], SCENE["objects"][A], SCENE["objects"][B]

    def edges(o):
        return {
            "left": o["x"] - o["w"]/2, "right": o["x"] + o["w"]/2,
            "top":  o["y"] - o["h"]/2, "bottom": o["y"] + o["h"]/2,
            "cx": o["x"], "cy": o["y"]
        }

    eT, eA, eB = edges(tgt), edges(oA), edges(oB)
    MIN_SIZE = 1.0
    if axis == "both":
        axis = "x"  # choose one; change to "y" if you prefer

    if axis == "x":
        left_anchor  = oA if eA["cx"] <= eT["cx"] else None
        right_anchor = oA if eA["cx"] >  eT["cx"] else None
        left_anchor  = left_anchor  or (oB if eB["cx"] <= eT["cx"] else None)
        right_anchor = right_anchor or (oB if eB["cx"] >  eT["cx"] else None)

        desired_left  = None if left_anchor  is None else snap_to_grid(edges(left_anchor)["right"])
        desired_right = None if right_anchor is None else snap_to_grid(edges(right_anchor)["left"])

        if desired_left is not None and desired_right is not None:
            gap = max(MIN_SIZE, desired_right - desired_left)
            tgt["w"] = gap                         # do NOT snap here
            tgt["x"] = desired_left + gap / 2.0    # do NOT snap here
        elif desired_left is not None:
            cx = eT["cx"]
            new_w = max(MIN_SIZE, 2.0 * (cx - desired_left))
            tgt["w"] = snap_to_grid(new_w)
            tgt["x"] = snap_to_grid(cx)
        elif desired_right is not None:
            cx = eT["cx"]
            new_w = max(MIN_SIZE, 2.0 * (desired_right - cx))
            tgt["w"] = snap_to_grid(new_w)
            tgt["x"] = snap_to_grid(cx)

    elif axis == "y":
        top_anchor    = oA if eA["cy"] <= eT["cy"] else None
        bottom_anchor = oA if eA["cy"] >  eT["cy"] else None
        top_anchor    = top_anchor    or (oB if eB["cy"] <= eT["cy"] else None)
        bottom_anchor = bottom_anchor or (oB if eB["cy"] >  eT["cy"] else None)

        desired_top    = None if top_anchor    is None else snap_to_grid(edges(top_anchor)["bottom"])
        desired_bottom = None if bottom_anchor is None else snap_to_grid(edges(bottom_anchor)["top"])

        if desired_top is not None and desired_bottom is not None:
            gap = max(MIN_SIZE, desired_bottom - desired_top)
            tgt["h"] = gap                         # do NOT snap here
            tgt["y"] = desired_top + gap / 2.0     # do NOT snap here
        elif desired_top is not None:
            cy = eT["cy"]
            new_h = max(MIN_SIZE, 2.0 * (cy - desired_top))
            tgt["h"] = snap_to_grid(new_h)
            tgt["y"] = snap_to_grid(cy)
        elif desired_bottom is not None:
            cy = eT["cy"]
            new_h = max(MIN_SIZE, 2.0 * (desired_bottom - cy))
            tgt["h"] = snap_to_grid(new_h)
            tgt["y"] = snap_to_grid(cy)

def engine_align(targets, axis, mode):
    tlist = [t.upper() for t in (targets or []) if t]
    objs = [SCENE["objects"][t] for t in tlist if t in SCENE["objects"]]
    if not objs: return
    if axis == "x":
        if mode == "centers":
            cx = snap_to_grid(sum(o["x"] for o in objs)/len(objs))
            for o in objs: o["x"] = cx
        elif mode == "lefts":
            lx = min(o["x"] - o["w"]/2 for o in objs)
            for o in objs: o["x"] = snap_to_grid(lx + o["w"]/2)
        elif mode == "rights":
            rx = max(o["x"] + o["w"]/2 for o in objs)
            for o in objs: o["x"] = snap_to_grid(rx - o["w"]/2)
    elif axis == "y":
        if mode == "centers":
            cy = snap_to_grid(sum(o["y"] for o in objs)/len(objs))
            for o in objs: o["y"] = cy
        elif mode == "tops":
            top = min(o["y"] - o["h"]/2 for o in objs)
            for o in objs: o["y"] = snap_to_grid(top + o["h"]/2)
        elif mode == "bottoms":
            bot = max(o["y"] + o["h"]/2 for o in objs)
            for o in objs: o["y"] = snap_to_grid(bot - o["h"]/2)

def engine_distribute(targets, axis, mode, spacing):
    tlist = [t.upper() for t in (targets or []) if t]
    objs = [SCENE["objects"][t] for t in tlist if t in SCENE["objects"]]
    if len(objs) < 3:
        return

    def left(o):   return o["x"] - o["w"]/2
    def right(o):  return o["x"] + o["w"]/2
    def top(o):    return o["y"] - o["h"]/2
    def bottom(o): return o["y"] + o["h"]/2

    if axis == "x":
        # Sort by current position left-to-right
        objs.sort(key=lambda o: o["x"])

        left_bound  = min(left(o) for o in objs)
        right_bound = max(right(o) for o in objs)
        total_w = sum(o["w"] for o in objs)
        gaps = len(objs) - 1

        if mode == "equal_gaps":
            # Equalize EDGE gaps while keeping extremes anchored
            avail = right_bound - left_bound - total_w
            gap = avail / gaps  # may be < 0; we still honor “equal” (can overlap)
            # If you prefer to avoid overlap, uncomment the next two lines:
            # gap = max(0.0, gap)
            # gap = snap_to_grid(gap)
            cur_left = left_bound
            for o in objs:
                o["x"] = snap_to_grid(cur_left + o["w"]/2)
                cur_left += o["w"] + gap

        elif mode == "fixed_spacing":
            # Fixed edge gap; leftmost stays where it is; sequence to the right
            g = max(0.0, float(spacing or 0.0))
            g = snap_to_grid(g)
            cur_left = left_bound
            for o in objs:
                o["x"] = snap_to_grid(cur_left + o["w"]/2)
                cur_left += o["w"] + g

    elif axis == "y":
        # Sort by current position top-to-bottom
        objs.sort(key=lambda o: o["y"])

        top_bound    = min(top(o) for o in objs)
        bottom_bound = max(bottom(o) for o in objs)
        total_h = sum(o["h"] for o in objs)
        gaps = len(objs) - 1

        if mode == "equal_gaps":
            avail = bottom_bound - top_bound - total_h
            gap = avail / gaps
            cur_top = top_bound
            for o in objs:
                o["y"] = snap_to_grid(cur_top + o["h"]/2)
                cur_top += o["h"] + gap

        elif mode == "fixed_spacing":
            g = max(0.0, float(spacing or 0.0))
            g = snap_to_grid(g)
            cur_top = top_bound
            for o in objs:
                o["y"] = snap_to_grid(cur_top + o["h"]/2)
                cur_top += o["h"] + g


def engine_scale(target: str, axis: str, factor: float):
    t = target.upper()
    if t not in SCENE["objects"] or factor is None:
        return
    if factor <= 0:
        return
    o = SCENE["objects"][t]
    if axis in ("x", "both"):
        new_w = snap_to_grid(max(GRID_STEP, o["w"] * float(factor)))
        o["w"] = new_w
    if axis in ("y", "both"):
        new_h = snap_to_grid(max(GRID_STEP, o["h"] * float(factor)))
        o["h"] = new_h
    # keep center x,y; snap already handled on sizes


def engine_render_svg(path, view, grid=True):
    assert view == "topdown"
    w_px, h_px = 800, 600
    sx = w_px / SCENE["grid_w"]
    sy = h_px / SCENE["grid_h"]
    def gx(x): return x * sx
    def gy(y): return y * sy

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w_px} {h_px}">',
        '  <defs>',
        '    <marker id="arrowhead" markerWidth="6" markerHeight="6" refX="5" refY="2" orient="auto" markerUnits="strokeWidth">',
        '      <path d="M0,0 L0,4 L6,2 z" fill="red" />',
        '    </marker>',
        '  </defs>'
    ]

    if grid:
        x = 0.0
        while x <= SCENE["grid_w"] + 1e-9:
            parts.append(f'<line x1="{gx(x):.1f}" y1="0" x2="{gx(x):.1f}" y2="{h_px}" stroke="#eee" stroke-width="1"/>')
            x += GRID_STEP
        y = 0.0
        while y <= SCENE["grid_h"] + 1e-9:
            parts.append(f'<line x1="0" y1="{gy(y):.1f}" x2="{w_px}" y2="{gy(y):.1f}" stroke="#eee" stroke-width="1"/>')
            y += GRID_STEP

    # pass 1: footprints + labels
    for lab, o in SCENE["objects"].items():
        prim = (o.get("primitive") or "cube").lower()
        if prim in {"cube", "plane"}:
            x = gx(o["x"]); y = gy(o["y"])
            w = gx(o["w"]) - gx(0); h = gy(o["h"]) - gy(0)
            rx = x - w/2; ry = y - h/2
            parts.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{w:.1f}" height="{h:.1f}" '
                         f'fill="#cccccc" stroke="#333" stroke-width="2"/>')
            parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="monospace" font-size="14" '
                         f'text-anchor="middle" dominant-baseline="middle">{lab}</text>')

        elif prim == "ramp":
            # Draw a rotated rectangle using yaw from rot_deg[2]
            cx, cy = float(o["x"]), float(o["y"])
            L, W = float(o["w"]), float(o["h"])
            hx, hy = 0.5 * L, 0.5 * W

            yaw_deg = float((o.get("rot_deg") or [0,0,0])[2])
            yaw = math.radians(yaw_deg)

            # local corners in (X,Y) where X=length, Y=width
            local = [(+hx,+hy), (+hx,-hy), (-hx,-hy), (-hx,+hy)]

            def rot_world(xl, yl):
                xr =  xl*math.cos(yaw) - yl*math.sin(yaw)
                yr =  xl*math.sin(yaw) + yl*math.cos(yaw)
                return gx(cx + xr), gy(cy + yr)  # convert to pixels

            pts = [rot_world(xl, yl) for (xl, yl) in local]
            path = "M " + " L ".join(f"{px:.1f},{py:.1f}" for (px, py) in pts) + " Z"
            parts.append(
                f'<path d="{path}" fill="#ddd" stroke="#b33" stroke-width="2" stroke-dasharray="6,4"/>'
            )

            # Center label
            parts.append(f'<text x="{gx(cx):.1f}" y="{gy(cy):.1f}" font-family="monospace" font-size="14" '
                         f'text-anchor="middle" dominant-baseline="middle">{lab}</text>')

        else:
            continue

    # pass 2: arrows
    for lab, o in SCENE["objects"].items():
        if (o.get("primitive") or "").lower() != "ramp_arrow":
            continue
        parent_name = o.get("parent")
        parent = SCENE["objects"].get(parent_name)
        if not parent or any(k not in parent for k in ("x","y","w","h")):
            continue

        px, py = gx(parent["x"]), gy(parent["y"])
        L_px = abs(gx(parent["w"]) - gx(0))

        rot_deg = parent.get("rot_deg") or [0,0,0]
        yaw = math.radians(float(rot_deg[2]))   # Z index (yaw)

        pad = 6.0
        half = max(0.0, 0.5 * (L_px * 0.8) - pad)
        dx = math.cos(yaw) * half
        dy = -math.sin(yaw) * half              # Y-down canvas

        dir_up = (o.get("dir","up") == "up")
        sx, sy = (px - dx, py - dy)
        ex, ey = (px + dx, py + dy)
        if not dir_up:
            sx, sy, ex, ey = ex, ey, sx, sy

        parts.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="red" stroke-width="3" marker-end="url(#arrowhead)"/>'
        )

    # CLOSE SVG and write once (always write; no DRY_RUN, no LAST_SVG_SUMMARY)
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return os.path.abspath(path)

def engine_export_state(path):
    with open(path, "w", encoding="utf-8") as f: json.dump(SCENE, f, indent=2)
    return os.path.abspath(path)
def _exists(label: str) -> bool:
    return bool(label) and label.upper() in SCENE["objects"]

def _edge_box(o):
    return {
        "left":  o["x"] - o["w"]/2, "right": o["x"] + o["w"]/2,
        "top":   o["y"] - o["h"]/2, "bottom": o["y"] + o["h"]/2,
        "x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"]
    }
EPS = GRID_STEP * 0.25  # ~12.5 cm tolerance for “already aligned”

def _aabb(o):
    """Return left, right, top, bottom for an object's footprint (top-down)."""
    x, y, w, h = o["x"], o["y"], o["w"], o["h"]
    return (x - w/2, x + w/2, y - h/2, y + h/2)

def _point_rect_min_dist(px: float, py: float, rect: tuple[float,float,float,float]) -> float:
    """Distance from point (px,py) to axis-aligned rectangle (l,r,t,b)."""
    l, r, t, b = rect
    # clamp point to rect
    cx = min(max(px, l), r)
    cy = min(max(py, t), b)
    dx = px - cx
    dy = py - cy
    return math.hypot(dx, dy)

def _union_intervals(intervals: list[tuple[float,float]]) -> list[tuple[float,float]]:
    """Union of [a,b] intervals on a line."""
    if not intervals: return []
    ints = sorted([(min(a,b), max(a,b)) for (a,b) in intervals])
    out = [ints[0]]
    for a,b in ints[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))  # merge
        else:
            out.append((a,b))
    return out

def _empty_spans_1d(occupied: list[tuple[float,float]], low: float, high: float) -> list[tuple[float,float]]:
    """Complement intervals (empty spans) within [low, high]."""
    if low > high: low, high = high, low
    occ = _union_intervals([(max(low,a), min(high,b)) for a,b in occupied if b > low and a < high])
    if not occ: return [(low, high)]
    spans = []
    cur = low
    for a,b in occ:
        if a > cur:
            spans.append((cur, a))
        cur = max(cur, b)
    if cur < high:
        spans.append((cur, high))
    return spans

def _scanline_empty_span_max(objs: dict, gw: float, gh: float) -> tuple[float, dict]:
    """
    Conservative LOS estimator (axis-aligned):
    - Horizontal scanlines at y = 0, edges (top/bottom) and centers of all boxes, and gh.
    - Vertical scanlines at x = 0, edges and centers, and gw.
    Returns (max_span, {"max_h":..., "max_v":..., "where":{...}}).
    """
    # collect scanlines only from footprinted rects
    ys = {0.0, gh}
    xs = {0.0, gw}
    rects = {}
    for L,o in objs.items():

        l,r,t,b = _aabb(o)
        rects[L] = (l,r,t,b)
        ys.update([t, b, o["y"]])
        xs.update([l, r, o["x"]])

    # horizontal
    max_h = 0.0; where_h = None
    for y in sorted(ys):
        occ = []
        for (l,r,t,b) in rects.values():
            if t <= y <= b:
                occ.append((l, r))
        empty = _empty_spans_1d(occ, 0.0, gw)
        for a,b in empty:
            span = b - a
            if span > max_h:
                max_h = span
                where_h = {"y": float(y), "x0": float(a), "x1": float(b)}

    # vertical
    max_v = 0.0; where_v = None
    for x in sorted(xs):
        occ = []
        for (l,r,t,b) in rects.values():
            if l <= x <= r:
                occ.append((t, b))
        empty = _empty_spans_1d(occ, 0.0, gh)
        for a,b in empty:
            span = b - a
            if span > max_v:
                max_v = span
                where_v = {"x": float(x), "y0": float(a), "y1": float(b)}

    return max(max_h, max_v), {"max_h": max_h, "where_h": where_h, "max_v": max_v, "where_v": where_v}

def _near(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps

# --- label helpers ---
def _exists_label(L: str) -> bool:
    return bool(L) and L.upper() in SCENE["objects"]

def _filter_existing(labels: list[str]) -> list[str]:
    return [L.upper() for L in (labels or []) if _exists_label(L)]
def _norm_label(s: str | None) -> str | None:
    return None if not s else str(s).strip().upper()

def _gensym(prefix: str = "P") -> str:
    """Generate a fresh one/two-letter label not in the scene."""
    base = prefix[:1].upper() if prefix else "P"
    # try single letters A..Z, then base+01..99
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        L = c if base == "P" else base + c
        if L not in SCENE["objects"]:
            return L
    for i in range(1, 100):
        L = f"{base}{i:02d}"
        if L not in SCENE["objects"]:
            return L
    raise RuntimeError("E_LABEL_SPACE: ran out of labels")

_ASM_DIR = {
    "left": "left_of",  "west": "left_of",
    "right": "right_of","east": "right_of",
    "up": "above",      "north":"above",  "front": "above",
    "down":"below",     "south":"below",  "back":  "below",
    "left_of":"left_of","right_of":"right_of","above":"above","below":"below"
}
def _snapf(v, minv=GRID_STEP):
    if v is None: return None
    return snap_to_grid(max(minv, float(v)))

# ---------- Phase 4: Anchors & Constraints helpers ----------

def _ensure_obj(label: str):
    return SCENE["objects"].get(label.upper())

def _edges(o):
    return {
        "left":  o["x"] - o["w"]/2, "right": o["x"] + o["w"]/2,
        "top":   o["y"] - o["h"]/2, "bottom": o["y"] + o["h"]/2,
        "cx": o["x"], "cy": o["y"], "w": o["w"], "h": o["h"]
    }

def _set_center(o, cx=None, cy=None):
    if cx is not None: o["x"] = snap_to_grid(cx)
    if cy is not None: o["y"] = snap_to_grid(cy)

def _set_left(o, x_left):
    o["x"] = snap_to_grid(x_left + o["w"]/2)

def _set_right(o, x_right):
    o["x"] = snap_to_grid(x_right - o["w"]/2)

def _set_top(o, y_top):
    o["y"] = snap_to_grid(y_top + o["h"]/2)

def _set_bottom(o, y_bottom):
    o["y"] = snap_to_grid(y_bottom - o["h"]/2)

# Anchors: persistent % from canvas
def engine_set_anchor(label: str, x_pct: float=None, y_pct: float=None):
    L = label.upper()
    if L not in SCENE["objects"]:
        return
    a = SCENE["anchors"].setdefault(L, {})
    if x_pct is not None: a["x_pct"] = float(x_pct)
    if y_pct is not None: a["y_pct"] = float(y_pct)

def engine_apply_anchors():
    # Re-apply percent anchors after grid resize
    gw, gh = SCENE["grid_w"], SCENE["grid_h"]
    for L, a in SCENE["anchors"].items():
        o = _ensure_obj(L)
        if not o: continue
        cx = a.get("x_pct")
        cy = a.get("y_pct")
        _set_center(o,
            cx * gw if cx is not None else None,
            cy * gh if cy is not None else None
        )
# Constraint schema (stored in SCENE["constraints"]):
# kind:
#   "align_left"         target, a
#   "align_right"        target, a
#   "align_centers_x"    target, a
#   "align_centers_y"    target, a
#   "between_x"          target, a, b           # center target between a/b on X
#   "between_y"          target, a, b
#   "edge_gap_x"         a, b, gap              # set edge gap between a->b along X
#   "edge_gap_y"         a, b, gap
# (You can add more later)

def engine_add_constraint(c: dict):
    # sanitize/upper
    c = dict(c)
    for k in ("target","a","b"):
        if k in c and c[k]: c[k] = c[k].upper()
    SCENE["constraints"].append(c)

def engine_remove_constraint(index: int):
    if 0 <= index < len(SCENE["constraints"]):
        del SCENE["constraints"][index]

def engine_clear_constraints():
    SCENE["constraints"].clear()

def _solve_once():
    # One deterministic pass; repeat a few times for propagation
    for c in SCENE["constraints"]:
        kind = c.get("kind")
        target = _ensure_obj(c.get("target")) if c.get("target") else None
        a = _ensure_obj(c.get("a")) if c.get("a") else None
        b = _ensure_obj(c.get("b")) if c.get("b") else None
        gap = float(c.get("gap") or 0.0)

        if kind == "align_left" and target and a:
            _set_left(target, _edges(a)["left"])

        elif kind == "align_right" and target and a:
            _set_right(target, _edges(a)["right"])

        elif kind == "align_centers_x" and target and a:
            _set_center(target, cx=_edges(a)["cx"])

        elif kind == "align_centers_y" and target and a:
            _set_center(target, cy=_edges(a)["cy"])

        elif kind == "between_x" and target and a and b:
            ea, eb = _edges(a), _edges(b)
            mid = snap_to_grid((ea["cx"] + eb["cx"]) / 2.0)
            _set_center(target, cx=mid)

        elif kind == "between_y" and target and a and b:
            ea, eb = _edges(a), _edges(b)
            mid = snap_to_grid((ea["cy"] + eb["cy"]) / 2.0)
            _set_center(target, cy=mid)

        elif kind == "edge_gap_x" and a and b:
            # place b to the right of a with edge gap 'gap'
            ea, eb = _edges(a), _edges(b)
            desired_left_b = ea["right"] + snap_to_grid(gap)
            _set_left(b, desired_left_b)

        elif kind == "edge_gap_y" and a and b:
            ea, eb = _edges(a), _edges(b)
            desired_top_b = ea["bottom"] + snap_to_grid(gap)
            _set_top(b, desired_top_b)

        elif kind == "align_tops" and target and a:
            _set_top(target, _edges(a)["top"])

        elif kind == "align_bottoms" and target and a:
            _set_bottom(target, _edges(a)["bottom"])

def engine_solve_constraints(iterations: int = 5):
    # Re-apply anchors first (if canvas size changed)
    engine_apply_anchors()
    for _ in range(iterations):
        _solve_once()

def _aligned(targets: list[str], axis: str, mode: str) -> bool:
    objs = [SCENE["objects"][t.upper()] for t in targets if _exists(t)]
    if len(objs) < 2:
        return False

    eb = [_edge_box(o) for o in objs]  # each has left,right,top,bottom,x,y

    # pick which coordinate to compare
    key = None
    if axis == "x":
        key = {"centers": "x", "lefts": "left", "rights": "right"}.get(mode)
    elif axis == "y":
        key = {"centers": "y", "tops": "top", "bottoms": "bottom"}.get(mode)
    if not key:
        return False

    ref = eb[0][key]
    return all(_near(e[key], ref) for e in eb[1:])


def _distributed_equal_gaps(targets: list[str], axis: str) -> bool:
    objs = [SCENE["objects"][t.upper()] for t in targets if _exists(t)]
    if len(objs) < 3: return False

    if axis == "x":
        objs.sort(key=lambda o: o["x"])
        gaps = []
        for i in range(len(objs)-1):
            right_i = objs[i]["x"] + objs[i]["w"]/2
            left_j  = objs[i+1]["x"] - objs[i+1]["w"]/2
            gaps.append(round(left_j - right_i, 6))
        return len(set(gaps)) == 1
    else:
        objs.sort(key=lambda o: o["y"])
        gaps = []
        for i in range(len(objs)-1):
            bottom_i = objs[i]["y"] + objs[i]["h"]/2
            top_j    = objs[i+1]["y"] - objs[i+1]["h"]/2
            gaps.append(round(top_j - bottom_i, 6))
        return len(set(gaps)) == 1

def _distributed_fixed_spacing(targets: list[str], axis: str, spacing: float) -> bool:
    objs = [SCENE["objects"][t.upper()] for t in targets if _exists(t)]
    if len(objs) < 3: return False

    if axis == "x":
        objs.sort(key=lambda o: o["x"])
        for i in range(len(objs)-1):
            right_i = objs[i]["x"] + objs[i]["w"]/2
            left_j  = objs[i+1]["x"] - objs[i+1]["w"]/2
            if abs((left_j - right_i) - float(spacing)) > 1e-6:
                return False
        return True
    else:
        objs.sort(key=lambda o: o["y"])
        for i in range(len(objs)-1):
            bottom_i = objs[i]["y"] + objs[i]["h"]/2
            top_j    = objs[i+1]["y"] - objs[i+1]["h"]/2
            if abs((top_j - bottom_i) - float(spacing)) > 1e-6:
                return False
        return True


def _move_is_noop(target: str, dx: float, dy: float) -> bool:
    o = SCENE["objects"].get(target.upper())
    return bool(o and abs(dx) < 1e-9 and abs(dy) < 1e-9)

def _scale_is_noop(target: str, axis: str, factor: float) -> bool:
    if abs(factor - 1.0) < 1e-9: return True
    return False
def _require_exists(label: str, code: str):
    if not _exists(label):
        raise ValueError(f"{code}: target '{label}' not found")

def _require_all_exist(labels: list[str], code: str):
    missing = [t for t in labels if not _exists(t)]
    if missing:
        raise ValueError(f"{code}: targets not found {missing}")

def engine_resize_canvas(new_w: int, new_h: int, scale_sizes: bool = False):
    """Resize canvas while preserving each object's relative position.
       If scale_sizes=True, also scale w/h proportionally."""
    old_w, old_h = SCENE["grid_w"], SCENE["grid_h"]
    if old_w <= 0 or old_h <= 0:
        SCENE["grid_w"], SCENE["grid_h"] = new_w, new_h
        return

    # 1) compute percents from the current absolute positions
    percents = {}
    for L, o in SCENE["objects"].items():
        px = o["x"] / float(old_w)
        py = o["y"] / float(old_h)
        percents[L] = (px, py)

    # 2) optionally compute size scale factors
    sx = new_w / float(old_w)
    sy = new_h / float(old_h)

    # 3) apply new canvas size
    SCENE["grid_w"], SCENE["grid_h"] = int(new_w), int(new_h)

    # 4) re-position (and optionally resize) objects
    for L, o in SCENE["objects"].items():
        px, py = percents[L]
        o["x"] = snap_to_grid(px * SCENE["grid_w"])
        o["y"] = snap_to_grid(py * SCENE["grid_h"])
        if scale_sizes:
            # Keep proportions (uniform), or you can choose non-uniform min(sx,sy)
            s = (sx + sy) / 2.0
            o["w"] = snap_to_grid(max(GRID_STEP, o["w"] * s))
            o["h"] = snap_to_grid(max(GRID_STEP, o["h"] * s))


# ---------------- Router ----------------
def route_and_execute(command_batch: Dict, natural: str = "", merge_existing: bool = False) -> Dict[str, str]:
    global _LAST_OBJECT_LABEL, LAST_REMOVED_BBOX, UNDO_STACK, REDO_STACK
    out = {}
    snap_taken = False
    def _maybe_snapshot():
        nonlocal snap_taken
        if not snap_taken:
            HISTORY.append({"scene": _deepcopy_scene()})
            UNDO_STACK.append(_deepcopy_scene())
            REDO_STACK.clear()
            snap_taken = True

    try:
        commands = command_batch["commands"]

        for item in commands:
            tool = item["tool"]
            args = dict(item.get("arguments") or {})

            # Normalize casing...
            if "labels" in args and args["labels"]:
                args["labels"] = [x.upper() for x in args["labels"]]
            for k in ("target","a","b"):
                if k in args and args[k]:
                    args[k] = args[k].upper()
            if "targets" in args and args["targets"]:
                args["targets"] = [x.upper() for x in args["targets"]]

            # --- Validation per tool ---
            def fail(code, msg):
                # Emit a single error and stop (handled by outer try/except)
                raise ValueError(f"{code}: {msg}")

            if tool == "report_error":
                # If the model sent a report_error tool call, surface it and stop.
                code = args.get("code") or "E_MODEL"
                msg  = args.get("message") or "model error"
                return {"error_code": code, "error_message": msg}

            # --- tool routing ---
            if tool == "create_scene":
                if not (args.get("labels") or args.get("count")):
                    raise ValueError("E_ARGS_CREATE: labels or count required")

                # only block reset if merge_existing is False
                if SCENE["objects"] and not (RESET_WORDS.search(natural or "") or merge_existing):
                    raise ValueError("E_EDIT_ONLY: scene not empty; say 'reset' to recreate")

                engine_create_scene(
                    labels=args.get("labels") or [],
                    primitive=args.get("primitive") or "cube",
                    count=int(args.get("count") or len(args.get("labels") or [])),
                    placement=args.get("placement") or "random_nonoverlap",
                    size=float(args.get("size") or 3.0),
                    margin=float(args.get("margin") or 0.8),
                    seed=int(args.get("seed") or 0),
                    grid_w=int(args.get("grid_w") or GRID_W),
                    grid_h=int(args.get("grid_h") or GRID_H),
                )
                engine_solve_constraints()

            elif tool == "resize_canvas":
                _maybe_snapshot()
                if args.get("grid_w") is None or args.get("grid_h") is None:
                    fail("E_ARGS_RESIZE", "grid_w and grid_h required")
                gw = int(args["grid_w"]); gh = int(args["grid_h"])
                if gw < 1 or gh < 1:
                    fail("E_ARGS_RESIZE", "grid_w/grid_h must be ≥ 1")

                scale_sizes = bool(args.get("scale_sizes", False))
                engine_resize_canvas(gw, gh, scale_sizes=scale_sizes)

                # Re-apply anchors & constraints so persistent relationships hold
                engine_solve_constraints()
            elif tool == "reset_scene":
                _reset_scene()         # or tool_reset_scene({}) if you prefer that wrapper
                continue               # ← keep executing the rest of the batch


            elif tool == "place_relative":
                _maybe_snapshot()

                # fallback: if target missing, use last created/edited object
                tgt = args.get("target") or _LAST_OBJECT_LABEL
                ref = args.get("ref")

                # accept both vocabularies:
                # - compass style via 'direction': east/west/north/south (also right/left/up/down/front/back)
                # - edge style via 'direction': left_of/right_of/above/below
                raw_dir = (args.get("direction") or "").lower()

                # normalize all synonyms to the edge-style set
                compass_to_rel = {
                    "east": "right_of", "right": "right_of",
                    "west": "left_of",  "left":  "left_of",
                    "north": "above",   "up":    "above", "front": "above",
                    "south": "below",   "down":  "below", "back":  "below",
                }
                rel = raw_dir if raw_dir in {"left_of","right_of","above","below"} else compass_to_rel.get(raw_dir)

                # distance (accepts distance or gap)
                dist = float(args.get("distance") or args.get("gap") or GRID_STEP)
                dist = snap_to_grid(max(0.0, dist))

                # validate
                if not tgt or not ref or rel not in {"left_of","right_of","above","below"}:
                    fail("E_ARGS_REL", "target, ref, direction required")

                _require_all_exist([tgt, ref], "E_NOT_FOUND")
                T = SCENE["objects"][tgt.upper()]
                R = SCENE["objects"][ref.upper()]
                eR = _edges(R)

                if rel == "left_of":
                    _set_right(T, eR["left"] - dist)
                elif rel == "right_of":
                    _set_left(T, eR["right"] + dist)
                elif rel == "above":
                    _set_bottom(T, eR["top"] - dist)
                elif rel == "below":
                    _set_top(T, eR["bottom"] + dist)

                engine_solve_constraints()

            elif tool == "stack_above":
                _maybe_snapshot()
                tgt = args.get("target"); ref = args.get("ref")
                g   = float(args.get("gap") or args.get("distance") or 0.0)
                center_xy = True if args.get("center") is None else bool(args.get("center"))
                if not tgt or not ref:
                    fail("E_ARGS_STACK", "target and ref required"); continue
                _require_exists(tgt, "E_NOT_FOUND"); _require_exists(ref, "E_NOT_FOUND")
                engine_stack_above(tgt, ref, g, center_xy=center_xy)
                engine_solve_constraints()

            elif tool == "stack_below":
                _maybe_snapshot()
                tgt = args.get("target"); ref = args.get("ref")
                g   = float(args.get("gap") or args.get("distance") or 0.0)
                center_xy = True if args.get("center") is None else bool(args.get("center"))
                if not tgt or not ref:
                    fail("E_ARGS_STACK", "target and ref required"); continue
                _require_exists(tgt, "E_NOT_FOUND"); _require_exists(ref, "E_NOT_FOUND")
                engine_stack_below(tgt, ref, g, center_xy=center_xy)
                engine_solve_constraints()


            elif tool == "place_above":
                _maybe_snapshot()
                tgt = args.get("target")
                new_label = args.get("new_label")
                gap = float(args.get("gap") or 0.0)
                if not tgt or not new_label:
                    fail("E_ARGS_PLACE_ABOVE", "target and new_label required")
                _require_exists(tgt, "E_NOT_FOUND")
                engine_place_above(tgt, new_label, gap=gap, copy_size=True)
                engine_solve_constraints()

            elif tool == "add_ramp":
                new_objs = tool_add_ramp(args, SCENE)
                SCENE["objects"].update(new_objs)
            elif tool == "move":
                _maybe_snapshot()
                if not args.get("target"):
                    fail("E_ARGS_MOVE", "target required")
                if args.get("dx") is None and args.get("dy") is None:
                    fail("E_ARGS_MOVE", "dx/dy required")
                _require_exists(args["target"], "E_NOT_FOUND")
                if _move_is_noop(args["target"], float(args.get("dx") or 0.0), float(args.get("dy") or 0.0)):
                    continue  # no-op
                                # If a list of targets is provided, use group move (with symmetry option)
                if args.get("targets"):
                    engine_move_group(
                        args["targets"],
                        dx=float(args.get("dx") or 0.0),
                        dy=float(args.get("dy") or 0.0),
                        symmetric=bool(args.get("symmetric") or False),
                        pivot=(args.get("pivot") or "grid_center")
                    )
                    engine_solve_constraints()
                    continue

                engine_move(args["target"], float(args.get("dx") or 0.0), float(args.get("dy") or 0.0))
                engine_solve_constraints()
            elif tool == "merge_objects":
                _maybe_snapshot()
                keep = args.get("keep"); rem = args.get("remove")
                if not keep or not rem:
                    fail("E_ARGS_MERGE", "keep and remove required")
                _require_all_exist([keep,rem], "E_NOT_FOUND")
                engine_merge_objects(keep, rem)
                engine_solve_constraints()
            elif tool == "remove_object":
                _maybe_snapshot()
                if not args.get("target"):
                    fail("E_ARGS_REMOVE", "target required")
                _require_exists(args["target"], "E_NOT_FOUND")
                engine_remove_object(args["target"])
                engine_solve_constraints()
            elif tool == "move_into_bbox":
                _maybe_snapshot()
                if not args.get("target"):
                    fail("E_ARGS_MIB", "target required")
                _require_exists(args["target"], "E_NOT_FOUND")
                # uses LAST_REMOVED_BBOX if bbox not provided
                engine_move_into_bbox(args["target"], args.get("bbox"))
                engine_solve_constraints()
            elif tool == "rename_object":
                _maybe_snapshot()
                old = args.get("target") or args.get("label")
                new = args.get("new_label")
                if not old or not new:
                    fail("E_ARGS_RENAME", "target(old) and new_label required")
                _require_exists(old, "E_NOT_FOUND")
                engine_rename_object(old, new)
                engine_solve_constraints()
            elif tool == "place_left_of":
                _maybe_snapshot()
                w = args.get("w"); h = args.get("h")
                if (w is None or h is None) and natural:
                    pw, ph = _parse_wh_from_text(natural)
                    if pw is not None and ph is not None:
                        if w is None: w = pw
                        if h is None: h = ph
                engine_place_left_of(
                    args.get("target"),
                    args.get("new_label"),
                    float(args.get("gap") or 0.0),
                    w=w, h=h
                )
                engine_solve_constraints()
            elif tool == "place_right_of":
                _maybe_snapshot()
                w = args.get("w"); h = args.get("h")
                if (w is None or h is None) and natural:
                    pw, ph = _parse_wh_from_text(natural)
                    if pw is not None and ph is not None:
                        if w is None: w = pw
                        if h is None: h = ph
                engine_place_right_of(
                    args.get("target"),
                    args.get("new_label"),
                    float(args.get("gap") or 0.0),
                    w=w, h=h
                )
                engine_solve_constraints()
            elif tool == "place_below":
                _maybe_snapshot()
                w = args.get("w"); h = args.get("h")
                if (w is None or h is None) and natural:
                    pw, ph = _parse_wh_from_text(natural)
                    if pw is not None and ph is not None:
                        if w is None: w = pw
                        if h is None: h = ph
                engine_place_below(
                    args.get("target"),
                    args.get("new_label"),
                    float(args.get("gap") or 0.0),
                    w=w, h=h
                )
                engine_solve_constraints()
            elif tool == "batch_rename":
                _maybe_snapshot()
                pairs = args.get("pairs")  # e.g. [["E","D"],["F","E"],["C","F"]]
                if not pairs: fail("E_ARGS_BRename","pairs required"); continue
                engine_batch_rename(pairs)
                engine_solve_constraints()
            elif tool == "mirror_object":
                _maybe_snapshot()
                tgt = args.get("target")
                if not tgt: fail("E_ARGS_MIRROR","target required"); continue
                engine_mirror_object(tgt, axis=(args.get("axis") or "x"), pivot=(args.get("pivot") or "grid_center"))
                engine_solve_constraints()

            elif tool == "align":
                _maybe_snapshot()
                tgts = args.get("targets") or []
                axis = args.get("axis")
                mode = args.get("mode")

                # Validate axis/mode first
                if axis not in ("x", "y"):
                    fail("E_ARGS_ALIGN", "axis must be x or y"); continue
                if mode not in ("centers","lefts","rights","tops","bottoms"):
                    fail("E_ARGS_ALIGN", "invalid mode"); continue

                # Normal case: >=2 targets -> do your existing align
                if len(tgts) >= 2:
                    _require_all_exist(tgts, "E_NOT_FOUND")
                    if _aligned(tgts, axis, mode):
                        continue
                    engine_align(tgts, axis, mode)
                    engine_solve_constraints()
                    continue

                # Auto-upgrade: 1 target + centers -> center to bounds instead of failing
                if len(tgts) == 1 and mode == "centers":
                    t = tgts[0]
                    _require_exists(t, "E_NOT_FOUND")
                    side = "center_y" if axis == "y" else "center_x"
                    # If you have engine_align_to_bounds, use it:
                    try:
                        engine_align_to_bounds(target=t, side=side, gap=0.0)
                    except NameError:
                        # Fallback: manual center to grid
                        if side == "center_x":
                            SCENE["objects"][t.upper()]["x"] = snap_to_grid(SCENE["grid_w"] / 2.0)
                        else:
                            SCENE["objects"][t.upper()]["y"] = snap_to_grid(SCENE["grid_h"] / 2.0)
                    engine_solve_constraints()
                    print(f"[WARN] align with 1 target auto-upgraded to {side} centering for {t}")
                    continue

                # Otherwise: 0 targets or non-centers single target -> skip (do not abort batch)
                print("[WARN] E_ARGS_ALIGN: need ≥2 targets (or single 'centers'); skipping align")
                continue

            elif tool == "distribute":
                _maybe_snapshot()
                if not args.get("targets") or len(args["targets"]) < 3:
                    fail("E_ARGS_DISTRIBUTE", "≥3 targets required")
                if args.get("axis") not in ("x", "y"):
                    fail("E_ARGS_DISTRIBUTE", "axis must be x or y")
                if args.get("mode") not in ("equal_gaps", "fixed_spacing"):
                    fail("E_ARGS_DISTRIBUTE", "invalid mode")
                if args["mode"] == "fixed_spacing" and args.get("spacing") is None:
                    fail("E_ARGS_DISTRIBUTE", "spacing required for fixed_spacing")
                _require_all_exist(args["targets"], "E_NOT_FOUND")
                if args["mode"] == "equal_gaps" and _distributed_equal_gaps(args["targets"], args["axis"]):
                    continue
                if args["mode"] == "fixed_spacing" and _distributed_fixed_spacing(args["targets"], args["axis"], float(args.get("spacing") or 0.0)):
                    continue
                engine_distribute(args["targets"], args["axis"], args["mode"], args.get("spacing"))
                engine_solve_constraints()
            elif tool == "scale":
                _maybe_snapshot()
                if not args.get("target"):
                    fail("E_ARGS_SCALE", "target required")

                ax = (args.get("axis") or "both")
                if ax not in ("x", "y", "both"):
                    fail("E_ARGS_SCALE", "axis must be x,y,or both")

                # --- NEW existence check ---
                _require_exists(args["target"], "E_NOT_FOUND")

                has_anchors = bool(args.get("a") and args.get("b"))
                if has_anchors:
                    _require_all_exist([args["a"], args["b"]], "E_NOT_FOUND")

                # Factor handling
                if args.get("factor") is None:
                    if has_anchors:
                        f = 1.0   # guard will adjust to exact touch
                    else:
                        fail("E_ARGS_SCALE", "factor required")
                else:
                    try:
                        f = float(args["factor"])
                    except Exception:
                        fail("E_ARGS_SCALE", "factor must be a number")
                    if f <= 0:
                        fail("E_ARGS_SCALE", "factor must be > 0")

                # --- NEW idempotence check (skip no-ops) ---
                if not has_anchors and _scale_is_noop(args["target"], ax, f):
                    continue

                # Apply scale
                engine_scale(args["target"], ax, f)

                # Guard correction if anchors exist
                if has_anchors:
                    guard_axis = ax if ax in ("x", "y") else "x"   # or "y" if you prefer
                    guard_scale_touch(args["target"], args["a"], args["b"], guard_axis)

                engine_solve_constraints()
            elif tool == "set_height":
                _maybe_snapshot()

                # accept absolute height or multiplicative factor
                h_abs = args.get("height")
                fac   = args.get("factor")

                if h_abs is None and fac is None:
                    fail("E_ARGS_HEIGHT", "height or factor required")
                    continue

                # determine target with fallback
                tgt = args.get("target") or _LAST_OBJECT_LABEL
                if tgt is None:
                    # no target and nothing in history; don't crash the batch
                    print("W_NO_TARGET", "no target and no last object; skipping set_height")
                    continue
                T = tgt.upper()
                if T not in SCENE["objects"]:
                    print("W_NOT_FOUND", f"target {T} not found; skipping set_height")
                    continue
                cur_h = SCENE["objects"][tgt.upper()]["height"]
                new_h = float(cur_h) * float(fac) if fac is not None else float(h_abs)
                engine_set_height(tgt, new_h)
                _LAST_OBJECT_LABEL = T  
                engine_solve_constraints()


            elif tool == "set_anchor":
                _maybe_snapshot()
                if not args.get("target"):
                    fail("E_ARGS_ANCHOR", "target required")
                _require_exists(args["target"], "E_NOT_FOUND")
                xp = args.get("x_pct"); yp = args.get("y_pct")
                if xp is None and yp is None:
                    fail("E_ARGS_ANCHOR", "x_pct and/or y_pct required")
                if xp is not None and not (0.0 <= float(xp) <= 1.0):
                    fail("E_ARGS_ANCHOR", "x_pct must be in [0,1]")
                if yp is not None and not (0.0 <= float(yp) <= 1.0):
                    fail("E_ARGS_ANCHOR", "y_pct must be in [0,1]")
                engine_set_anchor(args["target"], x_pct=xp, y_pct=yp)
                engine_solve_constraints()
            elif tool == "add_constraint":
                _maybe_snapshot()
                kind = args.get("kind")
                if not kind:
                    fail("E_ARGS_CONSTRAINT", "kind required")
                # basic validation by kind
                need = []
                if kind in ("align_left","align_right","align_centers_x","align_centers_y"):
                    need = ["target","a"]
                elif kind in ("between_x","between_y"):
                    need = ["target","a","b"]
                elif kind in ("edge_gap_x","edge_gap_y"):
                    need = ["a","b","gap"]
                elif kind in ("align_tops","align_bottoms"):
                    need = ["target","a"]
                for k in need:
                    if args.get(k) is None:
                        fail("E_ARGS_CONSTRAINT", f"{k} required for {kind}")

                # existence checks
                for k in ("target","a","b"):
                    if args.get(k): _require_exists(args[k], "E_NOT_FOUND")

                # dedupe identical constraints
                newc = {k: args.get(k) for k in ("kind","target","a","b","gap")}
                newc = {k:v for k,v in newc.items() if v is not None}
                if newc.get("kind") in ("between_x","between_y") and newc.get("target"):
                    k = newc["kind"]; tgt = newc["target"]
                    SCENE["constraints"] = [
                        c for c in SCENE["constraints"]
                        if not (c.get("kind")==k and c.get("target")==tgt)
                    ]
                # --- latest align_tops/bottoms wins per target ---
                if newc.get("kind") in ("align_tops","align_bottoms") and newc.get("target"):
                    if args.get("replace", True):
                        k = newc["kind"]; tgt = newc["target"]
                        SCENE["constraints"] = [
                            c for c in SCENE["constraints"]
                            if not (c.get("kind")==k and c.get("target")==tgt)
                        ]

                if newc not in SCENE["constraints"]:
                    engine_add_constraint(newc)
                    engine_solve_constraints()
            elif tool == "remove_constraint":
                _maybe_snapshot()
                if args.get("index") is None:
                    fail("E_ARGS_CONSTRAINT", "index required")
                engine_remove_constraint(int(args["index"]))
                engine_solve_constraints()
            elif tool == "clear_constraints":
                _maybe_snapshot()
                engine_clear_constraints()
                engine_solve_constraints()
            elif tool == "solve_constraints":
                _maybe_snapshot()
                engine_solve_constraints()
            elif tool == "undo":
                if not HISTORY:
                    fail("E_UNDO_EMPTY", "no history available")
                snap = HISTORY.pop()
                _restore_scene(snap["scene"])
                engine_solve_constraints()  # keep scene consistent after restore
            elif tool == "redo":
                engine_redo()
                engine_solve_constraints()
            elif tool == "render_svg":

                if args.get("view") != "topdown":
                    fail("E_VIEW_REQUIRED", "render_svg must use view='topdown'")
                seed = int(args.get("seed") or 0)
                svg_path = _next_artifact_path(seed, "svg")
                svg_path = engine_render_svg(path=svg_path, view="topdown", grid=bool(args.get("grid", True)))
                out["svg"] = svg_path
                  # log artifact even if export doesn't follow in this plan
                ARTIFACTS.append({"svg": svg_path})

            elif tool == "export_state":

                seed = int(args.get("seed") or 0)
                json_path = _next_artifact_path(seed, "json")
                json_path = engine_export_state(path=json_path)
                out["json"] = json_path
                  # attach json to the last artifact if it doesn't have one yet
                if ARTIFACTS and "json" not in ARTIFACTS[-1]:
                    ARTIFACTS[-1]["json"] = json_path
                else:
                    ARTIFACTS.append({"json": json_path})

            elif tool == "align_to_bounds":
                _maybe_snapshot()
                tgt = args.get("target"); side = args.get("side")
                if not tgt or side not in ("left","right","top","bottom","center_x","center_y"):
                    fail("E_ARGS_ATB","target+valid side required")
                _require_exists(tgt,"E_NOT_FOUND")
                o = SCENE["objects"][tgt.upper()]

                if side == "left":     _set_left(o,   GRID_STEP/2)
                elif side == "right":  _set_right(o,  SCENE["grid_w"] - GRID_STEP/2)
                elif side == "top":    _set_top(o,    GRID_STEP/2)
                elif side == "bottom": _set_bottom(o, SCENE["grid_h"] - GRID_STEP/2)
                elif side == "center_x": _set_center(o, cx=SCENE["grid_w"]/2)
                elif side == "center_y": _set_center(o, cy=SCENE["grid_h"]/2)

                engine_solve_constraints()
            elif tool == "align_to_ref":
                _maybe_snapshot()
                tgt, ref, edge = args.get("target"), args.get("ref"), args.get("edge")
                gap = float(args.get("gap") or 0.0)
                if not tgt or not ref or not edge:
                    fail("E_ARGS_ATR","target, ref, edge required")
                _require_all_exist([tgt,ref],"E_NOT_FOUND")
                T, R = SCENE["objects"][tgt.upper()], SCENE["objects"][ref.upper()]
                eR = _edges(R); gap = snap_to_grid(gap)

                if edge == "left_to_right":   _set_left(T,  eR["right"] + gap)
                elif edge == "right_to_left": _set_right(T, eR["left"]  - gap)
                elif edge == "left_to_left":  _set_left(T,  eR["left"]  + gap)
                elif edge == "right_to_right":_set_right(T, eR["right"] - gap)
                elif edge == "top_to_bottom": _set_top(T,   eR["bottom"]+ gap)
                elif edge == "bottom_to_top": _set_bottom(T,eR["top"]   - gap)
                elif edge == "top_to_top":    _set_top(T,   eR["top"]   + gap)
                elif edge == "bottom_to_bottom": _set_bottom(T, eR["bottom"] - gap)
                elif edge == "center_x":      _set_center(T, cx=eR["cx"])
                elif edge == "center_y":      _set_center(T, cy=eR["cy"])

                engine_solve_constraints()
            
            elif tool == "add_object":
                _maybe_snapshot()
                if not args.get("label"):
                    fail("E_ARGS_ADD", "label required")
                if _exists_label(args["label"]):
                    return {"error_code": "E_DUPLICATE_LABEL", "error_message": "label already exists"}
                prim = args.get("primitive") or "cube"
                if prim not in ("cube","square","rect"):
                    fail("E_ARGS_ADD", "invalid primitive")
                if args.get("w") is not None and args["w"] < GRID_STEP:
                    args["w"] = GRID_STEP
                if args.get("h") is not None and args["h"] < GRID_STEP:
                    args["h"] = GRID_STEP
                engine_add_object(
                    label=args["label"],
                    primitive=prim,
                    x=args.get("x"),
                    y=args.get("y"),
                    w=args.get("w") or (args.get("size") or 1.0),
                    h=args.get("h"),
                    margin=float(args.get("margin") or 0.8),
                    height=args.get("height")
                )
                engine_solve_constraints()
            else:
                fail("E_TOOL_UNKNOWN", f"Unknown tool: {tool}")

        return out

    except ValueError as e:
        # Standardize router failure as a clean error result
        msg = str(e)
        if ": " in msg:
            code, rest = msg.split(": ", 1)
        else:
            code, rest = "E_ROUTER", msg
        return {"error_code": code.strip(), "error_message": rest.strip()}



# ---------------- Tiny rule-based agent ----------------
PRIMITIVES = ["cube","square","rect"]

def _extract_count(text: str) -> Optional[int]:
    m = re.search(r"\b(\d+)\b", text); return int(m.group(1)) if m else None

def _extract_labels(text: str) -> List[str]:
    m = re.search(r"(?:labels?|labeled as|labelled as)\s*[:\-]?\s*([A-Za-z,\s]+)", text, re.I)
    if not m: return []
    tokens = re.findall(r"[A-Za-z]+", m.group(1))
    one_letter = [t.upper() for t in tokens if len(t) == 1]
    seen, out = set(), []
    for t in one_letter:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _extract_primitive(text: str) -> str:
    for p in PRIMITIVES:
        if re.search(rf"\b{p}s?\b", text, re.I): return "rect" if p=="rect" else p
    if re.search(r"\b(square|cube)\b", text, re.I): return "square"
    return "cube"

def _extract_placement(text: str) -> str:
    if re.search(r"\brandom\b", text, re.I): return "random_nonoverlap"
    if re.search(r"\bgrid\b", text, re.I): return "grid"
    if re.search(r"\brow\b", text, re.I): return "row"
    if re.search(r"\bcluster\b", text, re.I): return "cluster"
    return "random_nonoverlap"

def ask_agent_multi(natural: str, model: str = "gpt-4o") -> Dict:
    """
    Uses Chat Completions with Structured Outputs (JSON Schema).
    Returns: {"commands": [...]} exactly matching TOOL_PLAN_SCHEMA.
    """
    # Make sure SYSTEM_PROMPT exists in your notebook (the big rules string).
    seed = prompt_seed(natural)

    system = (
        (SYSTEM_PROMPT if 'SYSTEM_PROMPT' in globals() else "")
        + f"\n\nDerived SEED for this request: {seed}\n"
           "- Always include this seed in every tool call's arguments.\n"
          "- Always end with render_svg(view='topdown') and export_state.\n"
          "- For MOVE: require target + (dx,dy) or (direction + distance).\n"
          "- For ALIGN: require targets + axis + mode.\n"
          "- For DISTRIBUTE: require targets + axis + mode; spacing only if mode=fixed_spacing."
    )
    scene_summary = json.dumps(build_scene_summary(natural), ensure_ascii=False)
    comp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "system", "content": f"SCENE_STATE (read-only JSON):\n{scene_summary}"},
            {"role": "user", "content": natural},
        ],
        response_format={"type": "json_schema", "json_schema": TOOL_PLAN_SCHEMA},
    )

    raw = comp.choices[0].message.content  # JSON string per schema
    plan = json.loads(raw)

    # Safety: enforce seed + topdown view for every command
    for cmd in plan.get("commands", []):
        args = cmd.setdefault("arguments", {})
        args.setdefault("seed", seed)
        if cmd["tool"] == "render_svg":
            args["view"] = "topdown"
    # Enforce seed/view + repair missing edit args from the prompt
    plan = _repair_plan(plan, natural, seed)
    return plan

def build_scene_summary(natural: str, max_items: int = 20) -> dict:
    # pick labels mentioned in the prompt (plus their neighbors if needed)
    labels = set(re.findall(r"\b[A-Za-z]\b", natural))
    labels = {L.upper() for L in labels}
    objs = SCENE["objects"]
    keys = list(objs.keys())

    # if no labels found, include up to max_items to keep context small
    chosen = [k for k in keys if k in labels] or keys[:max_items]

    def pack(o):
        x, y, w, h = o["x"], o["y"], o["w"], o["h"]
        return {
            "x": x, "y": y, "w": w, "h": h,
            "left":  x - w/2, "right": x + w/2,
            "top":   y - h/2, "bottom": y + h/2
        }

    return {
        "grid_w": SCENE["grid_w"],
        "grid_h": SCENE["grid_h"],
        "grid_step": GRID_STEP,
        "objects": {k: pack(objs[k]) for k in chosen if k in objs}
    }


# ---------------- Runner ----------------
def _normalize_editor_model(model: dict) -> dict:
    """Normalize editor JSON into engine scene format."""
    if not isinstance(model, dict):
        return {"grid_w": GRID_W, "grid_h": GRID_H, "objects": {}}
    grid_w = float(model.get("grid_w") or (model.get("canvas") or {}).get("width_m") or GRID_W)
    grid_h = float(model.get("grid_h") or (model.get("canvas") or {}).get("height_m") or GRID_H)
    objs = model.get("objects") or {}
    if isinstance(objs, list):
        out = {}
        for o in objs:
            if not all(k in o for k in ("x", "y", "w", "h")):
                continue
            lab = str(o.get("label") or o.get("id") or "OBJ").upper()
            item = {
                "primitive": str(o.get("primitive") or "cube"),
                "x": float(o["x"]), "y": float(o["y"]),
                "w": float(o["w"]), "h": float(o["h"]),
                "height": float(o.get("height", 0.5)),
            }
            if "z_offset" in o:
                item["z_offset"] = float(o["z_offset"])
            out[lab] = item
        objs = out
    else:
        objs = {str(k).upper(): v for k, v in dict(objs).items()}
    return {"grid_w": grid_w, "grid_h": grid_h, "objects": objs}


def _load_scene_from_model(model: dict | None) -> None:
    """Replace global SCENE from an editor model before running agent."""
    if not model:
        return
    state = _normalize_editor_model(model)
    _restore_scene(state)


def run_prompt(prompt: str, model: str | None = None, base_model: dict | None = None) -> dict:
    """
    Takes current scene JSON (base_model), merges agent edits into it.
    Returns {"svg": "...", "json": "..."}.
    """
    # load current scene if provided
    if base_model:
        try:
            _load_scene_from_model(base_model)
        except Exception as e:
            print("Failed to load base_model:", e)

    # interpret user command
    commands = ask_agent_multi(prompt, model=(model or OPENAI_AGENT_MODEL))

    # execute and update existing scene rather than replacing it
    outputs = route_and_execute(commands, prompt, merge_existing=True)

    return {k: v for k, v in outputs.items() if k in ("svg", "json")}





import re

DIR_TO_DXY = {
    "left":  (-1.0, 0.0),
    "right": ( 1.0, 0.0),
    "up":    ( 0.0,-1.0),  # top-down screen coords
    "down":  ( 0.0, 1.0),
}

def _labels_from_text(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]+", text)
    out, seen = [], set()
    for t in tokens:
        if len(t) == 1:
            T = t.upper()
            if T not in seen:
                seen.add(T); out.append(T)
    return out

def _clamp_nonneg(v: Optional[float], default=None):
    if v is None:
        return default
    try:
        return max(0.0, float(v))
    except:
        return default

def _clamp_min(v: Optional[float], minv: float, default=None):
    if v is None:
        return default if default is not None else minv
    try:
        return max(minv, float(v))
    except:
        return minv

def _repair_plan(plan: dict, natural: str, seed: int) -> dict:
    """
    Ensures required args exist for move/align/distribute and fills view/seed.
    Uses regex to recover missing values from the original prompt.
    """
     # --- Undo intent detection (before per-command edits) ---
    # If the user asks to undo/revert/go back and the plan doesn't already contain 'undo',
    # replace the plan with: undo -> render -> export.
    if re.search(r"\b(undo|reverse|revert|go back|back)\b", natural, re.I):
        has_undo = any(c.get("tool") == "undo" for c in plan.get("commands", []))
        if not has_undo:
            plan["commands"] = [
                {"tool": "undo", "arguments": {"seed": seed}},
                {"tool": "render_svg", "arguments": {"seed": seed, "view": "topdown"}},
                {"tool": "export_state", "arguments": {"seed": seed}}
            ]
            return plan
        # --- "between" intent -> add_constraint between_x / between_y ---
    m_between = re.search(r"\bbetween\s+([A-Za-z])\s+(?:and|&)\s+([A-Za-z])\b", natural, re.I)
    m_target  = re.search(r"\b(?:cube|square|rect)?\s*([A-Za-z])\b.*?\bbetween\b", natural, re.I)
    # decide axis: default horizontal (x) unless the user says vertical/above/below/y
    axis_between = "x"
    if re.search(r"\b(vertical|y[-\s]*axis|above|below)\b", natural, re.I):
        axis_between = "y"

    if m_between and m_target:
        tgt = m_target.group(1).upper()
        a   = m_between.group(1).upper()
        b   = m_between.group(2).upper()
        # ensure target exists if they just said "put cube Z ..."
        ensure = []
        if tgt not in SCENE["objects"]:
            ensure.append({"tool":"ensure_object","arguments":{"label":tgt,"primitive":"cube","size":1.0}})
        plan["commands"] = ensure + [
            {"tool":"add_constraint","arguments":{
                "kind": f"between_{axis_between}",
                "target": tgt, "a": a, "b": b
            }},
            {"tool":"solve_constraints","arguments":{}},
            {"tool":"render_svg","arguments":{"seed":seed,"view":"topdown"}},
            {"tool":"export_state","arguments":{"seed":seed}}
        ]
        return plan
# --- 0) Pre-scan: collect labels that will be created later in this plan
    future_new_labels = []
    for c in plan.get("commands", []):
        if c.get("tool") in ("add_object", "ensure_object"):
            lab = ((c.get("arguments") or {}).get("label") or "").upper()
            if lab:
                future_new_labels.append(lab)

    last_new_label = None  # track the most recently added/ensured label as we walk commands

    for cmd in plan.get("commands", []):
        tool = cmd.get("tool", "")
        args = cmd.setdefault("arguments", {})
    # If this command creates an object, remember its label
    if tool in ("add_object", "ensure_object"):
        lab = (args.get("label") or "").upper()
        if lab:
            last_new_label = lab

    # Sanitize single-slot references, but don't nuke labels that will be created later in this plan
    for k in ("target", "a", "b"):
        val = (args.get(k) or "")
        if val:
            V = val.upper()
            if not _exists_label(V) and V not in future_new_labels:
                if tool != "add_object":
                    args[k] = None

    # Fallback for place_relative: if target missing, bind to the last created label
    if tool == "place_relative" and not args.get("target") and last_new_label:
        args["target"] = last_new_label

            # --- Truth checks to kill hallucinated/nonexistent labels ---
        # sanitize single slots
        for k in ("target","a","b"):
            if args.get(k) and not _exists_label(args[k]):
                # If a tool requires an existing object, scrub to force error or no-op;
                # if it's add_object, we'll handle below.
                if tool != "add_object":
                    args[k] = None
        # sanitize lists
        if args.get("targets"):
            args["targets"] = _filter_existing(args["targets"])

        # --- Touch anchor extraction for scale ---
        if cmd.get("tool") == "scale":
            # If user wrote "touch X and Y" (or "touch X, Y"), fill a/b if missing
            if (not args.get("a") or not args.get("b")) and re.search(r"\btouch\b", natural, re.I):
                m = re.search(r"\btouch\s+([A-Za-z])(?:\s*(?:and|,)\s*([A-Za-z]))?", natural, re.I)
                if m:
                    if not args.get("a"): args["a"] = m.group(1).upper()
                    if not args.get("b") and m.group(2): args["b"] = m.group(2).upper()

            # If axis missing, infer from wording
            if not args.get("axis"):
                if re.search(r"\b(x[-\s]*axis|horizontally|left|right)\b", natural, re.I):
                    args["axis"] = "x"
                elif re.search(r"\b(y[-\s]*axis|vertically|top|bottom)\b", natural, re.I):
                    args["axis"] = "y"
                else:
                    args["axis"] = "y"  # default if ambiguous

            # If user said "touch" but no factor provided, let GPT omit factor.
            # Router will apply engine_scale with whatever factor GPT produced;
            # the guard above will correct to exact touch using anchors if needed.

        args.setdefault("seed", seed)

        if tool == "render_svg":
            args["view"] = "topdown"
        if "spacing" in args:
            args["spacing"] = _clamp_nonneg(args.get("spacing"), default=0.0)
        if "margin" in args:
            args["margin"] = _clamp_nonneg(args.get("margin"), default=0.8)

        # Size-ish
        if "size" in args:
            args["size"] = _clamp_min(args.get("size"), GRID_STEP, default=GRID_STEP)
        if "w" in args:
            args["w"] = _clamp_min(args.get("w"), GRID_STEP, default=None)
        if "h" in args:
            args["h"] = _clamp_min(args.get("h"), GRID_STEP, default=None)
        if "height" in args:
            args["height"] = _clamp_min(args.get("height"), GRID_STEP, default=None)

        if tool == "move":
            # target
            if not args.get("target"):
                m = re.search(r"\bmove\s+([A-Za-z])\b", natural, re.I)
                if m: args["target"] = m.group(1).upper()
            # dx/dy (either "by dx,dy" or "dir dist")
            if args.get("dx") is None and args.get("dy") is None:
                m = re.search(r"\bmove\s+[A-Za-z]\s+by\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)", natural, re.I)
                if m:
                    args["dx"] = float(m.group(1)); args["dy"] = float(m.group(2))
                else:
                    m = re.search(r"\bmove\s+[A-Za-z]\s+(left|right|up|down)\s+([0-9.]+)", natural, re.I)
                    if m:
                        d = m.group(1).lower(); dist = float(m.group(2))
                        sx, sy = DIR_TO_DXY[d]
                        args["dx"] = sx * dist; args["dy"] = sy * dist
        if tool == "add_object":

            if not args.get("label"):
                m = re.search(r"\blabel(?:ed|led)?\s+([A-Za-z])\b", natural, re.I)
                if m:
                    args["label"] = m.group(1).upper()
            if re.search(r"\bcenter\b", natural, re.I):
                want_center = False
                if args.get("x") is None and args.get("y") is None:
                    want_center = True
                else:
                    # treat 0/0 as "not really set" when user asked for center
                    xv = args.get("x"); yv = args.get("y")
                    if (xv is None or abs(float(xv)) < 1e-9) and (yv is None or abs(float(yv)) < 1e-9):
                        want_center = True
                if want_center and "SCENE" in globals():
                    args["x"] = SCENE["grid_w"] / 2.0
                    args["y"] = SCENE["grid_h"] / 2.0
            # primitive default
            if not args.get("primitive"):
                # accept wording like "cube", "square", "rect"
                m = re.search(r"\b(cube|square|rect|rectangle)\b", natural, re.I)
                if m:
                    kind = m.group(1).lower()
                    args["primitive"] = "rect" if kind in ("rect","rectangle") else kind
                else:
                    args["primitive"] = "cube"
            if args.get("height") is None:
                m = re.search(r"\bheight\s*(?:scaled\s*to|=)?\s*([0-9]+(?:\.[0-9]+)?)\b", natural, re.I)
                if m:
                    args["height"] = float(m.group(1))
            # size: parse "2x2", "2 x 2", or "size 2x2"
            if args.get("w") is None and args.get("h") is None and not args.get("size"):
                m = re.search(r"\b(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\b", natural, re.I)
                if m:
                    args["w"] = float(m.group(1))
                    args["h"] = float(m.group(2))
            # if only "size N" is present, make it square
            if args.get("size") is not None:
                if args.get("w") is None and args.get("h") is None:
                    args["w"] = float(args["size"])
                    args["h"] = float(args["size"])
            txt = natural.lower()
             # If the prompt says "cube" or "square", enforce square top-down unless an explicit W×H was given.
            is_squareish = bool(re.search(r"\b(cube|square)\b", txt))
            explicit_wh  = bool(re.search(r"\b\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?\b", txt))

            # If user wrote "height ..." that refers to 3D extrusion only; do NOT leak into 2D h.
            # Heuristic fix: if 'height' is present and the plan set h == height by mistake, restore 2D h.
            if args.get("height") is not None:
                # If an explicit "size N" exists and no explicit W×H, prefer a square N×N.
                if args.get("size") is not None and not explicit_wh:
                    args.setdefault("w", float(args["size"]))
                    args.setdefault("h", float(args["size"]))
                # If 'cube' or 'square' was requested and no explicit W×H, enforce w==h.
                if is_squareish and not explicit_wh:
                    # If only w is known, mirror to h; if only h is known and it's equal to height, fix it.
                    if args.get("w") is not None and args.get("h") is None:
                        args["h"] = float(args["w"])
                    elif args.get("h") is not None and args.get("w") is None:
                        # If h accidentally equals the 3D height, reset it to a sane square using size or 1.0
                        if abs(float(args["h"]) - float(args["height"])) < 1e-9:
                            side = float(args.get("size") or 1.0)
                            args["w"] = side
                            args["h"] = side
                    elif args.get("w") is not None and args.get("h") is not None:
                        # If plan made h == 3D height, snap back to square using w
                        if abs(float(args["h"]) - float(args["height"])) < 1e-9:
                            args["h"] = float(args["w"])

            # Final guard: if they said "cube" and neither w/h/size were explicit, keep a square 1×1 (snapped later)
            if is_squareish and not explicit_wh and args.get("w") is None and args.get("h") is None and args.get("size") is None:
                args["w"] = 1.0
                args["h"] = 1.0
            lab = (args.get("label") or "").upper()
            if lab:
                create_if_missing = bool(re.search(r"\b(create if missing)\b", natural, re.I))
                if _exists_label(lab) and create_if_missing:
                    i = 2
                    base = lab
                    while _exists_label(f"{base}{i}"):
                        i += 1
                    args["label"] = f"{base}{i}"
        if tool == "align":
            # align A C E vertically centers | horizontally tops
            if not args.get("targets"):
                m = re.search(r"\balign\s+([A-Za-z,\s]+?)\s+(vertically|horizontally)", natural, re.I)
                if m: args["targets"] = _labels_from_text(m.group(1))
            if not args.get("axis"):
                m = re.search(r"\balign\s+[A-Za-z,\s]+?\s+(vertically|horizontally)", natural, re.I)
                if m: args["axis"] = "x" if m.group(1).lower() == "vertically" else "y"
            if not args.get("mode"):
                m = re.search(r"\b(centers|lefts|rights|tops|bottoms)\b", natural, re.I)
                if m: args["mode"] = m.group(1).lower()
            # 1) Parse "top/bottom/left/right ... of H and Z" phrasing if targets not set
            if not args.get("targets"):
                m = re.search(r"\b(top|bottom|left|right)s?\s+edges?\s+of\s+([A-Za-z](?:\s*(?:,|and)\s*[A-Za-z])*)", natural, re.I)
                if m:
                    args["mode"] = {"top":"tops","bottom":"bottoms","left":"lefts","right":"rights"}[m.group(1).lower()]
                    args["targets"] = _labels_from_text(m.group(2))

            # 2) If mode implies an axis, FORCE the correct axis (override contradictions)
            if args.get("mode") in ("tops","bottoms"):
                args["axis"] = "y"
            elif args.get("mode") in ("lefts","rights"):
                args["axis"] = "x"

            # 3) If only "centers" is mentioned, infer axis from words (fallback)
            if (args.get("mode") == "centers") and not args.get("axis"):
                if re.search(r"\b(horizontally|x[-\s]*axis|left|right)\b", natural, re.I):
                    args["axis"] = "x"
                elif re.search(r"\b(vertically|y[-\s]*axis|top|bottom)\b", natural, re.I):
                    args["axis"] = "y"
        if tool == "distribute":
            # distribute A C E horizontally equal gaps | spacing 2
            if not args.get("targets"):
                m = re.search(r"\bdistribute\s+([A-Za-z,\s]+?)\s+(vertically|horizontally)", natural, re.I)
                if m: args["targets"] = _labels_from_text(m.group(1))
            if not args.get("axis"):
                m = re.search(r"\bdistribute\s+[A-Za-z,\s]+?\s+(vertically|horizontally)", natural, re.I)
                if m: args["axis"] = "x" if m.group(1).lower() == "horizontally" else "y"
            if not args.get("mode"):
                m = re.search(r"\bequal\s+gaps?\s+of\s+([0-9.]+)", natural, re.I)
                if m:
                    args["mode"] = "fixed_spacing"
                    args["spacing"] = float(m.group(1))
                elif re.search(r"\bequal\s+gaps?\b", natural, re.I):
                    args["mode"] = "equal_gaps"
                else:
                    m = re.search(r"\bspacing\s+([0-9.]+)\b", natural, re.I)
                    if m:
                        args["mode"] = "fixed_spacing"
                        args["spacing"] = float(m.group(1))
    has_render = any(c.get("tool") == "render_svg" for c in plan.get("commands", []))
    has_export = any(c.get("tool") == "export_state" for c in plan.get("commands", []))
    if not has_render:
        plan["commands"].append({"tool": "render_svg", "arguments": {"seed": seed, "view": "topdown"}})
    if not has_export:
        plan["commands"].append({"tool": "export_state", "arguments": {"seed": seed}})
     # --- Phase 4: anchor extraction & insertion ---
    if re.search(r"\b(keep|always|anchor|pin)\b", natural, re.I):
        mL = re.search(r"\b(keep|always|anchor|pin)\s+([A-Za-z])\b", natural, re.I)
        if mL:
            label = mL.group(2).upper()
            mP = re.search(r"\bat\s*([0-9.]+)\s*%?\s*[, ]\s*([0-9.]+)\s*%?\b", natural, re.I)
            if mP:
                def _norm(v: str) -> float:
                    val = float(v)
                    return val/100.0 if val > 1.0 else val
                x_pct = _norm(mP.group(1))
                y_pct = _norm(mP.group(2))
            elif re.search(r"\b(center|centre|centered)\b", natural, re.I):
                x_pct = 0.5; y_pct = 0.5
            else:
                if "SCENE" in globals() and label in SCENE["objects"]:
                    o = SCENE["objects"][label]
                    x_pct = o["x"]/SCENE["grid_w"]; y_pct = o["y"]/SCENE["grid_h"]
                else:
                    x_pct = 0.5; y_pct = 0.5

            already = any(
                (cmd.get("tool") == "set_anchor" and
                 (cmd.get("arguments") or {}).get("target","").upper() == label)
                for cmd in plan.get("commands", [])
            )
            if not already:
                plan["commands"].insert(0, {
                    "tool": "set_anchor",
                    "arguments": {"target": label, "x_pct": x_pct, "y_pct": y_pct}
                })
    return plan



# ---- New Cell ----

# --- Blender MCP socket sender + delta applier for "objects" JSON schema

import json, socket, os, copy, time
from pathlib import Path

MCP_HOST = "localhost"
MCP_PORT = 9876
STATE_FILE = Path(".atlas_last_scene_state.json")  # persists across cells

def _send_to_blender_mcp(payload: dict, host=MCP_HOST, port=MCP_PORT, timeout=5):
    """Send a JSON command batch to the already-open Blender MCP add-on."""
    data = (json.dumps(payload) + "\n").encode("utf-8")  # newline-terminated
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(data)
        # (Optional) read a short reply if the add-on responds
        try:
            s.shutdown(socket.SHUT_WR)
            reply = s.recv(65536)
            if reply:
                print("MCP reply:", reply.decode("utf-8", errors="ignore")[:500])
        except Exception:
            pass

def _load_objects_from_json(json_path: str) -> dict:
    """Return the 'objects' dict; if file has 'commands' only, synthesize objects from add_object commands."""
    with open(json_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    if "objects" in spec and isinstance(spec["objects"], dict):
        return copy.deepcopy(spec["objects"])
    # fallback: build objects from add_object commands (best-effort)
    objs = {}
    for c in spec.get("commands", []):
        if c.get("tool") == "add_object":
            a = c.get("arguments", {}) or {}
            label = a.get("label") or a.get("name") or f"Obj{len(objs)+1}"
            objs[label] = {
                "primitive": a.get("primitive", "cube"),
                "x": a.get("x", 0.0), "y": a.get("y", 0.0), "z": a.get("z", 0.0),
                "w": a.get("w") or a.get("width") or a.get("size") or 1.0,
                "h": a.get("h") or a.get("depth") or a.get("size") or 1.0,
                "height": a.get("height") or a.get("size") or 1.0,
            }
    return objs

def _norm(obj: dict):
    """Normalize numeric fields to floats and provide defaults."""
    return {
        "primitive": obj.get("primitive", "cube"),
        "x": float(obj.get("x", 0.0)),
        "y": float(obj.get("y", 0.0)),
        "z": float(obj.get("z", 0.0)),
        "w": float(obj.get("w") or obj.get("width") or obj.get("size") or 1.0),
        "h": float(obj.get("h") or obj.get("depth") or obj.get("size") or 1.0),
        "height": float(obj.get("height") or obj.get("size") or 1.0),
    }

def _build_delta_commands(prev_objs: dict, curr_objs: dict):
    """
    Compare prev vs curr 'objects' dicts. Emit:
      - add_object for new labels
      - move for position deltas
      - resize (via scale-by-dimension) for dimension deltas
    """
    commands = []

    # 1) Adds for new objects
    for label, o in curr_objs.items():
        if label not in prev_objs:
            oN = _norm(o)
            commands.append({
                "tool": "add_object",
                "arguments": {
                    "label": label,
                    "primitive": oN["primitive"],
                    "x": oN["x"], "y": oN["y"], "z": oN["z"],
                    "w": oN["w"], "h": oN["h"], "height": oN["height"],
                }
            })

    # 2) Moves & resizes for existing objects
    def _ne(a,b,eps=1e-6): return abs(a-b) > eps

    for label, oCurr in curr_objs.items():
        if label not in prev_objs:
            continue
        a = _norm(prev_objs[label]); b = _norm(oCurr)

        # position delta
        dx, dy, dz = b["x"]-a["x"], b["y"]-a["y"], b["z"]-a["z"]
        if _ne(dx,0.0) or _ne(dy,0.0) or _ne(dz,0.0):
            commands.append({
                "tool": "move",
                "arguments": {"label": label, "dx": dx, "dy": dy, "dz": dz}
            })

        # size delta (emit a "resize_to" custom op that your MCP add-on should understand; if not, we can fall back to delete+add)
        if _ne(a["w"], b["w"]) or _ne(a["h"], b["h"]) or _ne(a["height"], b["height"]):
            commands.append({
                "tool": "resize_to",
                "arguments": {"label": label, "w": b["w"], "h": b["h"], "height": b["height"]}
            })

    # 3) (Optional) Deletes for removed objects
    for label in prev_objs.keys() - curr_objs.keys():
        commands.append({"tool": "delete", "arguments": {"label": label}})

    return commands

def _load_prev_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_state(objs: dict):
    STATE_FILE.write_text(json.dumps(objs, indent=2), encoding="utf-8")

def apply_scene_delta(json_path: str, reset_if_empty=True):
    """Compute delta from last state and push only the edits to Blender via MCP."""
    curr = _load_objects_from_json(json_path)

    # Normalize all
    currN = {k:_norm(v) for k,v in curr.items()}
    prevN = _load_prev_state()

    # First run? Do a reset + full add to ensure clean slate.
    if not prevN and reset_if_empty:
        cmds = []
        cmds.append({"tool": "reset_scene", "arguments": {}})  # let add-on clear the scene, set meters, etc.
        for label, o in currN.items():
            cmds.append({
                "tool":"add_object",
                "arguments":{"label":label,"primitive":o["primitive"],
                             "x":o["x"],"y":o["y"],"z":o["z"],
                             "w":o["w"],"h":o["h"],"height":o["height"]}
            })
        payload = {"commands": cmds}
        _send_to_blender_mcp(payload)
        _save_state(currN)
        print(f"Applied initial scene: {len(cmds)-1} objects.")
        return

    # Subsequent runs: compute delta
    cmds = _build_delta_commands(prevN, currN)
    if not cmds:
        print("No changes detected.")
        return

    payload = {"commands": cmds}
    _send_to_blender_mcp(payload)
    _save_state(currN)
    print(f"Applied {len(cmds)} change(s).")



def tool_add_ramp(args: dict, scene: dict):
    """
    Modes:
      - between: {"mode":"between","from":"A","to":"B"}
      - side:    {"mode":"side","of":"A","side":"left|right|front|back","length":<m?>,"slope_ratio":12}
    Requires: GRID_STEP, snap_to_grid, and _expand_ramp to exist.
    """
    objs = scene["objects"]
    mode = (args.get("mode") or "").lower()

    # -------- local helpers (no external deps) --------
    def _label(s): return (s or "").upper()

    def _top_height(o: dict) -> float:
        # top = center_z + height/2 ; defaults guard if fields missing
        cz = float(o.get("z_offset", 0.0))
        H  = float(o.get("height", 0.0))
        return cz + 0.5 * H

    def _footprint(o: dict):
        x = float(o.get("x", 0.0)); y = float(o.get("y", 0.0))
        w = float(o.get("w", 1.0)); h = float(o.get("h", 1.0))
        return x, y, w, h

    def _overlap(a0, a1, b0, b1):   # length of interval intersection
        return max(0.0, min(a1, b1) - max(a0, b0))

    def _closest_edge_pair_xy(A: dict, B: dict):
        """
        Choose the best facing edges and return (axis, ax, ay, bx, by)
        where 'axis' ∈ {'x','y'} and (ax,ay) is on A's edge, (bx,by) on B's.
        Always returns a 5-tuple.
        """
        Ax, Ay, Aw, Ah = _footprint(A)
        Bx, By, Bw, Bh = _footprint(B)

        AxL, AxR = Ax - Aw*0.5, Ax + Aw*0.5
        AyT, AyB = Ay - Ah*0.5, Ay + Ah*0.5
        BxL, BxR = Bx - Bw*0.5, Bx + Bw*0.5
        ByT, ByB = By - Bh*0.5, By + Bh*0.5

        ov_x = _overlap(AxL, AxR, BxL, BxR)   # widths overlap
        ov_y = _overlap(AyT, AyB, ByT, ByB)   # depths overlap
        dx   = abs(Bx - Ax)
        dy   = abs(By - Ay)
        EPS  = 1e-9

        # prefer axis where orthogonal overlap exists; otherwise by larger separation
        choose_x = False
        choose_y = False
        if ov_y > EPS and ov_x <= EPS:
            choose_x = True
        elif ov_x > EPS and ov_y <= EPS:
            choose_y = True
        elif ov_x > EPS and ov_y > EPS:
            choose_x = dx >= dy
            choose_y = not choose_x
        else:
            choose_x = dx >= dy
            choose_y = not choose_x

        if choose_x:
            # ramp runs along X: use facing vertical edges; keep endpoints EXACT (no snap)
            if Bx >= Ax:
                ax = AxR  # A right
                bx = BxL  # B left
            else:
                ax = AxL  # A left
                bx = BxR  # B right
            cy = (Ay + By) * 0.5
            return "x", ax, cy, bx, cy

        # else choose_y
        if By >= Ay:
            ay = AyB  # A bottom
            by = ByT  # B top
        else:
            ay = AyT  # A top
            by = ByB  # B bottom
        cx = (Ax + Bx) * 0.5
        return "y", cx, ay, cx, by

    # --------------- modes ---------------
    if mode == "between":
        a = _label(args.get("from"))
        b = _label(args.get("to"))
        if a not in objs or b not in objs or a == b:
            raise ValueError("E_NOT_FOUND: bad ramp endpoints")

        A, B = objs[a], objs[b]

        axis, ax, ay, bx, by = _closest_edge_pair_xy(A, B)

        # Width = overlap on orthogonal axis; fallback to min dimension if no overlap
        if axis == "x":
            # overlap in Y
            AyT, AyB = A["y"] - A["h"]*0.5, A["y"] + A["h"]*0.5
            ByT, ByB = B["y"] - B["h"]*0.5, B["y"] + B["h"]*0.5
            ov = _overlap(AyT, AyB, ByT, ByB)
            W = ov if ov > 0 else min(float(A["h"]), float(B["h"]))
        else:
            # overlap in X
            AxL, AxR = A["x"] - A["w"]*0.5, A["x"] + A["w"]*0.5
            BxL, BxR = B["x"] - B["w"]*0.5, B["x"] + B["w"]*0.5
            ov = _overlap(AxL, AxR, BxL, BxR)
            W = ov if ov > 0 else min(float(A["w"]), float(B["w"]))

        # Clamp/snap width only; keep endpoints exact so they touch
        W = max(GRID_STEP, snap_to_grid(W))

        # If edges coincide, nudge by one grid step to avoid L=0
        if axis == "x" and abs(bx - ax) < 1e-9:
            bx = ax + (GRID_STEP if B["x"] >= A["x"] else -GRID_STEP)
        if axis == "y" and abs(by - ay) < 1e-9:
            by = ay + (GRID_STEP if B["y"] >= A["y"] else -GRID_STEP)

        H0 = _top_height(A)
        H1 = _top_height(B)

        name = (args.get("label") or f"RAMP_{a}_{b}")
        k = 1
        while name in objs:
            name = f"{name}_{k:02d}"; k += 1

        return _expand_ramp(
            name,
            (ax, ay), (bx, by),
            W, H0, H1,
            thickness=0.1
        )


    elif mode == "side":
        a = _label(args.get("of"))
        side = (args.get("side") or "left").lower()
        if a not in objs:
            raise ValueError("E_NOT_FOUND: base block not found")

        A = objs[a]
        Ax, Ay, Aw, Ah = _footprint(A)
        H0 = _top_height(A)

        slope_ratio   = float(args.get("slope_ratio", 12.0))
        target_length = float(args.get("length", snap_to_grid(Aw * 2.0)))
        target_length = max(GRID_STEP, snap_to_grid(target_length))

        if side == "left":
            x0, y0 = snap_to_grid(Ax - Aw*0.5), snap_to_grid(Ay)
            x1, y1 = snap_to_grid(x0 - target_length), y0
        elif side == "right":
            x0, y0 = snap_to_grid(Ax + Aw*0.5), snap_to_grid(Ay)
            x1, y1 = snap_to_grid(x0 + target_length), y0
        elif side == "front":  # toward -Y
            x0, y0 = snap_to_grid(Ax), snap_to_grid(Ay - Ah*0.5)
            x1, y1 = x0, snap_to_grid(y0 - target_length)
        else:  # back
            x0, y0 = snap_to_grid(Ax), snap_to_grid(Ay + Ah*0.5)
            x1, y1 = x0, snap_to_grid(y0 + target_length)

        L = math.hypot(x1 - x0, y1 - y0)
        rise = L / max(1e-6, slope_ratio)
        H1   = H0 + rise

        # side width = block depth along orthogonal axis
        W = max(GRID_STEP, snap_to_grid(Ah if side in ("left","right") else Aw))

        base = args.get("label") or f"RAMP_{a}_{side}"
        name = base; k = 2
        while name in objs:
            name = f"{base}_{k:02d}"; k += 1

        return _expand_ramp(name, (x0, y0), (x1, y1), W, H0, H1, thickness=0.1)

    else:
        raise ValueError("E_ARGS: mode must be 'between' or 'side'")


# Import a JSON from outsource
# ==== UNIVERSAL PLAN JSON IMPORTER ===========================================
# Converts ANY reasonable top-down plan JSON into your agent's command batch:
# - Auto-detects units & scale; supports px/cm/m (+ optional scale bar in meta)
# - Finds "objects" anywhere; supports rect/circle/ellipse/polygon/polyline/line
# - Everything -> axis-aligned rectangle footprint; preserves label & height if present
# - Snaps all dimensions/positions to your 0.5 m grid
# - Finishes with render_svg + export_state
#
# Usage:
#   outs = import_any_topdown_json_and_build(plan_json_str, write_to_watch=True)
#   print(outs.get("svg"), outs.get("json"))

import math, json, os

GRID_CELL = 0.5     # your system's snap size (meters)
DEFAULT_H = 0.5     # fallback height if none is given (meters)
# ---- Pixel-faithful profile toggles ----
SNAP_POSITIONS = True     # snap centers to grid
SNAP_SIZES = False             # do NOT snap sizes
REBASE_TO_MARGIN = False    # no re-base; keep original px origin
MARGIN_M = 0.0
AUTO_CANVAS_POLICY = "resize_scene"  # critical
MAX_GRID_W = 400.0
MAX_GRID_H = 400.0
# Keep real heights; no plan cap by default (set to e.g. 0.20 if you want)
PLAN_MAX_H = None

# Accept only these shapes (unknowns skipped unless they provide x,y,w,h)
ACCEPTED_TYPES = {"rect", "circle", "ellipse", "polygon", "polyline"}
def _snap_pos(v, step=GRID_CELL):
    return round(float(v)/step)*step if SNAP_POSITIONS else float(v)

def _snap_size(v, step=GRID_CELL):
    return round(float(v)/step)*step if SNAP_SIZES else float(v)

def _canvas_px(plan, default=(800, 800)):
    cv = plan.get("canvas") or {}
    if "width" in cv and "height" in cv:
        return float(cv["width"]), float(cv["height"])
    vb = cv.get("viewBox") or cv.get("viewbox")
    if isinstance(vb, (list, tuple)) and len(vb) == 4:
        return float(vb[2]), float(vb[3])
    return default
def _label_from(entry):
    o, k = entry["node"], entry["key"]
    # preserve verbatim (no uppercasing)
    return str(o.get("label") or o.get("id") or k or "OBJ")

def _apply_height_cap(h):
    return min(h, PLAN_MAX_H) if PLAN_MAX_H else h


def _looks_like_room(ow, ia):
    if not (ow and ia): 
        return False
    try:
        xO,yO,wO,hO = float(ow["x"]), float(ow["y"]), float(ow["w"]), float(ow["h"])
        xI,yI,wI,hI = float(ia["x"]), float(ia["y"]), float(ia["w"]), float(ia["h"])
    except Exception:
        return False
    inside = (xO < xI) and (yO < yI) and (xO+wO > xI+wI) and (yO+hO > yI+hI)
    min_thick = max(1e-6, min(wO, hO) * 0.02)  # 2% thickness floor
    thick_ok  = (wO - wI) > min_thick and (hO - hI) > min_thick
    return inside and thick_ok

def _snap(v, step=GRID_CELL): 
    return round(float(v)/step)*step

def _units_to_m_per_px(units: str) -> float:
    """
    Returns meters per "pixel-like" unit. If units are physical, returns scale accordingly.
    - 'm'   -> 1.0 (already meters)
    - 'cm'  -> 0.01
    - 'mm'  -> 0.001
    - 'px'  -> derived separately from scale bar, else default 100 px ≈ 1 m
    - unknown -> assume px (100 px ≈ 1 m)
    """
    if not units: 
        # treat as px; m/px resolved later
        return None
    u = str(units).strip().lower()
    if u in ("m", "meter", "meters"):
        return 1.0
    if u in ("cm", "centimeter", "centimeters"):
        return 0.01
    if u in ("mm", "millimeter", "millimeters"):
        return 0.001
    if u in ("px", "pixel", "pixels"):
        return None  # defer to scale bar/default
    # unknown → treat as px
    return None

def _m_per_px_from_meta(plan: dict) -> float:
    """Use meta.approx_scale_bar.length_px ≈ 1 m if present, else default 100 px/m."""
    meta = (plan.get("meta") or {})
    sb = (meta.get("approx_scale_bar") or {})
    if isinstance(sb, dict) and "length_px" in sb:
        try:
            px = float(sb["length_px"])
            if px > 0:
                return 1.0 / px
        except:
            pass
    # default: 100 px ≈ 1 m
    return 1.0 / 100.0

def _resolve_scale(plan: dict):
    """
    Returns (m_per_unit, is_px_like).

    Heuristic:
    - If many nodes look like agent objects (have 'primitive' and 'height' and x,y,w,h),
      assume meters (1.0) and not px-like.
    - Else use explicit units if present, else fall back to px-like via meta/defaults.
    """
    # quick probe for agent-style objects
    def _probe_agentish(nodes):
        hits = 0; total = 0
        for n in nodes:
            if isinstance(n, dict):
                total += 1
                if all(k in n for k in ("x","y","w","h")) and ("primitive" in n) and ("height" in n):
                    hits += 1
        return hits, total

    # collect a shallow sample of dict values from plan
    sample_vals = []
    if isinstance(plan, dict):
        sample_vals.extend(v for v in plan.values() if isinstance(v, dict))
        obj = plan.get("objects")
        if isinstance(obj, dict):
            sample_vals.extend(list(obj.values())[:20])
        elif isinstance(obj, list):
            sample_vals.extend([v for v in obj if isinstance(v, dict)][:20])

    hits, total = _probe_agentish(sample_vals)
    if total and hits/total >= 0.4:   # 40% look agent-style → assume meters
        return 1.0, False

    # else use declared units / px-like path
    units = (plan.get("units") or plan.get("unit") or "").strip().lower()
    m_per_unit = _units_to_m_per_px(units)
    if m_per_unit is not None:
        return m_per_unit, False
    return _m_per_px_from_meta(plan), True

def _bbox_from_rect(o):
    # rect with top-left x,y and w,h
    return float(o["x"]), float(o["y"]), float(o["w"]), float(o["h"])

def _bbox_from_circle(o):
    cx, cy, r = float(o["cx"]), float(o["cy"]), float(o["r"])
    x = cx - r; y = cy - r; w = 2*r; h = 2*r
    return x, y, w, h

def _bbox_from_ellipse(o):
    cx, cy, rx, ry = float(o["cx"]), float(o["cy"]), float(o["rx"]), float(o["ry"])
    x = cx - rx; y = cy - ry; w = 2*rx; h = 2*ry
    return x, y, w, h

def _bbox_from_points(points):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    x0, y0 = min(xs), min(ys); x1, y1 = max(xs), max(ys)
    return x0, y0, (x1 - x0), (y1 - y0)

def _parse_points_any(o):
    """
    Accepts:
      - list of [x,y] pairs in o["points"]
      - SVG-like "points" string "x1,y1 x2,y2 ..."
      - explicit bbox dict in o["bbox"] = {x,y,w,h}
    """
    if "bbox" in o and isinstance(o["bbox"], dict):
        b = o["bbox"]; 
        if all(k in b for k in ("x","y","w","h")):
            return float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
    pts = o.get("points")
    if isinstance(pts, str):
        try:
            arr = []
            for token in pts.strip().split():
                xy = token.split(",")
                if len(xy) == 2:
                    arr.append((float(xy[0]), float(xy[1])))
            if arr:
                return _bbox_from_points(arr)
        except:
            return None
    if isinstance(pts, list) and pts and isinstance(pts[0], (list, tuple)):
        try:
            return _bbox_from_points([(float(a), float(b)) for a,b in pts])
        except:
            return None
    return None

def _bbox_from_line(o):
    # line: (x1,y1)→(x2,y2). thickness from strokeWidth if any (else small).
    x1, y1, x2, y2 = float(o["x1"]), float(o["y1"]), float(o["x2"]), float(o["y2"])
    stroke = float(o.get("strokeWidth", o.get("stroke_width", 0.0)) or 0.0)
    x0, x1m = min(x1,x2), max(x1,x2)
    y0, y1m = min(y1,y2), max(y1,y2)
    w = max(1e-6, x1m - x0); h = max(1e-6, y1m - y0)
    # inflate minimally by stroke to avoid zero-height rectangles
    w = max(w, stroke); h = max(h, stroke)
    return x0, y0, w, h
def _bbox_from_generic(o):
    t = (o.get("type") or o.get("shape") or "").strip().lower()

    try:
        if t == "rect":
            return float(o["x"]), float(o["y"]), float(o["w"]), float(o["h"])
        if t == "circle":
            cx, cy, r = float(o["cx"]), float(o["cy"]), float(o["r"])
            d = 2.0 * r
            return (cx - r), (cy - r), d, d
        if t == "ellipse":
            cx, cy, rx, ry = float(o["cx"]), float(o["cy"]), float(o["rx"]), float(o["ry"])
            return (cx - rx), (cy - ry), 2.0*rx, 2.0*ry
        if t in ("polygon", "polyline"):
            return _parse_points_any(o)
    except Exception:
        return None

    # Agent-style: center-based if primitive present
    if "primitive" in o and all(k in o for k in ("x","y","w","h")):
        try:
            return _bbox_from_agent_center(o)
        except Exception:
            return None

    # Fallback: top-left bbox if explicitly provided
    if all(k in o for k in ("x","y","w","h")):
        try:
            return float(o["x"]), float(o["y"]), float(o["w"]), float(o["h"])
        except Exception:
            return None

    return None

def _bbox_from_agent_center(o):
    cx, cy = float(o["x"]), float(o["y"])
    w, h   = float(o["w"]), float(o["h"])
    return cx - w/2.0, cy - h/2.0, w, h


def _extract_height(o):
    """
    Preserve any height-like info if present (meters if plan units were meters; else scaled).
    Checks common keys: height, z, z_height, H, thickness, elevation (as height of object, not base).
    """
    for k in ("height","z_height","H","thickness","z","elevation"):
        if k in o:
            try:
                return float(o[k])
            except:
                continue
    return None

def _collect_objects_anywhere(plan: dict):
    """
    Returns (structurals, items, details) with key hints preserved.
    - structurals: things under plan["room"] like outer_wall/inner_area/window/door (used to build wall bands)
    - items: plan["objects"] (furniture etc.)
    - details: tiny extras we still want (e.g., door leaf)
    Each entry is dict: {"node": o, "key": key_hint}
    """
    structurals, items, details = [], [], []

    def visit(node, key_hint=None, in_room=False, in_objects=False):
        if isinstance(node, dict):
            # Consider as drawable if it looks like a shape
            looks_like_shape = (node.get("type") or
                                any(k in node for k in ("x","y","w","h","cx","cy","r","rx","ry","points","x1","y1","x2","y2","bbox")))
            if looks_like_shape:
                entry = {"node": node, "key": key_hint}
                if in_room:
                    structurals.append(entry)
                elif in_objects:
                    items.append(entry)
                else:
                    details.append(entry)
            for k, v in node.items():
                visit(v, key_hint=k, in_room=in_room or (key_hint=="room"), in_objects=in_objects or (key_hint=="objects"))
        elif isinstance(node, list):
            for v in node:
                visit(v, key_hint=key_hint, in_room=in_room, in_objects=in_objects)

    # Prefer explicit sections
    if isinstance(plan.get("objects"), list):
        for o in plan["objects"]:
            items.append({"node": o, "key": o.get("id") or o.get("label") or "OBJ"})
    if isinstance(plan.get("room"), dict):
        visit(plan["room"], key_hint="room", in_room=True)

    # Light sweep for details (but not to duplicate items/room we already got)
    visit(plan)

    # Dedup by object id
    def _dedup(lst):
        seen = set(); out=[]
        for e in lst:
            oid = id(e["node"])
            if oid not in seen:
                seen.add(oid); out.append(e)
        return out
    return _dedup(structurals), _dedup(items), _dedup(details)

def _canvas_size(plan, m_per_unit, is_px_like, scene_bb_m=None, margin_m=1.0):
    """
    Decide grid_w/h (meters).
    - If canvas exists, use it but ensure it is big enough to include scene_bb_m + margins.
    - If no canvas or too small, auto-fit to scene bounds + margins.
    - Snap to GRID_CELL.
    scene_bb_m: (minx, miny, maxx, maxy) in meters (already scaled)
    """
    def _snap_up(v):  # snap outward so bounds fit
        return math.ceil(float(v)/GRID_CELL)*GRID_CELL

    want_w = want_h = None
    if scene_bb_m:
        minx, miny, maxx, maxy = scene_bb_m
        raw_w = max(0.5, (maxx - minx) + 2*margin_m)
        raw_h = max(0.5, (maxy - miny) + 2*margin_m)
        want_w, want_h = _snap_up(raw_w), _snap_up(raw_h)

    # Try to read incoming canvas
    canvas = plan.get("canvas") or {}
    W = H = None
    if all(k in canvas for k in ("width","height")):
        W = float(canvas["width"]); H = float(canvas["height"])
        if is_px_like:
            W, H = W * m_per_unit, H * m_per_unit
        W, H = _snap_up(W), _snap_up(H)
    else:
        vb = canvas.get("viewBox") or canvas.get("viewbox")
        if isinstance(vb, (list, tuple)) and len(vb) == 4:
            _, _, wv, hv = [float(x) for x in vb]
            W, H = (wv * m_per_unit if is_px_like else wv), (hv * m_per_unit if is_px_like else hv)
            W, H = _snap_up(W), _snap_up(H)

    # Decision: prefer provided canvas, but expand if needed; else use want_w/h or defaults
    if W is not None and H is not None:
        if want_w is not None and want_h is not None:
            W = max(W, want_w); H = max(H, want_h)
        return _snap_up(W), _snap_up(H)

    if want_w is not None and want_h is not None:
        return want_w, want_h

    # Last resort if we have nothing
    return 8.0, 8.0
def _height_m_from_obj(o: dict, m_per_unit: float, is_px_like: bool, default: float = DEFAULT_H) -> float:
    """
    Returns height in meters without snapping.
    Rules:
      - If the object declares a height unit, use it.
      - Else, if the overall plan is px-like:
          * assume heights are already meters unless they look 'pixel-ish' ( > 10 )
      - Else, pass through.
      - Clamp to a tiny positive epsilon so rugs etc. survive.
    """
    h = _extract_height(o)
    if h is None:
        return default

    # explicit per-object height unit hint
    units_h = (o.get("height_units") or o.get("units_height") or "").strip().lower()
    if units_h in ("m", "meter", "meters"):
        pass  # already meters
    elif units_h in ("cm", "centimeter", "centimeters"):
        h = float(h) / 100.0
    elif units_h in ("mm", "millimeter", "millimeters"):
        h = float(h) / 1000.0
    elif units_h in ("px", "pixel", "pixels"):
        h = float(h) * float(m_per_unit)
    else:
        # No explicit unit for height:
        # If geometry is px-like, treat small values as meters (typical furniture < 5 m)
        # and only scale if it looks like a pixel count (e.g., 30, 60, 120…).
        if is_px_like and float(h) > 10.0:
            h = float(h) * float(m_per_unit)

    return max(0.005, float(h))  # tiny epsilon, but *no snapping*

def _resolve_canvas_and_scale(plan):
    """
    Decide px vs m canvas. Return (desired_w_m, desired_h_m, m_per_unit, is_px_like, src).
    """
    cv = plan.get("canvas") or {}
    src = "px"

    if "width_m" in cv and "height_m" in cv:
        W, H = float(cv["width_m"]), float(cv["height_m"])
        src = "m"
    else:
        if "width" in cv and "height" in cv:
            W, H = float(cv["width"]), float(cv["height"])
        else:
            vb = cv.get("viewBox") or cv.get("viewbox")
            if isinstance(vb, (list, tuple)) and len(vb) == 4:
                W, H = float(vb[2]), float(vb[3])
            else:
                W, H = 800.0, 800.0

    m_per_unit, is_px_like = _resolve_scale(plan)
    if m_per_unit is None:
        m_per_unit, is_px_like = _m_per_px_from_meta(plan), True

    if src == "m":
        desired_w_m, desired_h_m = W, H
    else:
        desired_w_m, desired_h_m = W * m_per_unit, H * m_per_unit

    return desired_w_m, desired_h_m, m_per_unit, is_px_like, src
# === Live state & writer (put near the top of your Python file) ===

# Live state & writer (keep where you already placed it)
WATCH_PATH = r"C:\ATLAS\live_scene.json"   # MUST match blender_livesync.py
STATE = {
    "grid_w": 40.0,
    "grid_h": 30.0,
    "objects": {}
}

def _write_watch():
    os.makedirs(os.path.dirname(WATCH_PATH), exist_ok=True)
    with open(WATCH_PATH, "w", encoding="utf-8") as f:
        json.dump(STATE, f, ensure_ascii=False, indent=2)

def update_object_position(label, x_m, y_m):
    a = STATE["objects"].get(label)
    if not a:
        return
    a["x"] = _snap(x_m, GRID_CELL)
    a["y"] = _snap(y_m, GRID_CELL)
    _write_watch()

def update_object_size(label, w_m, h_m, anchor="center"):
    a = STATE["objects"].get(label)
    if not a:
        return
    cx, cy = float(a["x"]), float(a["y"])
    w_old, h_old = float(a.get("w", 1.0)), float(a.get("h", 1.0))

    W = max(_snap(w_m, GRID_CELL), GRID_CELL * 0.5)
    H = max(_snap(h_m, GRID_CELL), GRID_CELL * 0.5)

    dx = dy = 0.0
    if anchor in ("nw", "w", "sw"): dx = (W - w_old) / 2.0
    if anchor in ("ne", "e", "se"): dx = -(W - w_old) / 2.0
    if anchor in ("nw", "n", "ne"): dy = (H - h_old) / 2.0
    if anchor in ("sw", "s", "se"): dy = -(H - h_old) / 2.0

    a["w"], a["h"] = W, H
    a["x"], a["y"] = _snap(cx + dx, GRID_CELL), _snap(cy + dy, GRID_CELL)
    _write_watch()

def update_object_height(label, h_m):
    a = STATE["objects"].get(label)
    if not a:
        return
    a["height"] = max(_snap(h_m, GRID_CELL), 0.01)
    _write_watch()

def any_topdown_json_to_agent_batch(plan: dict,
                                    label_prefix: str = "",
                                    fit_to_scene: bool = True,      # ignored here
                                    margin_m: float = 0.0,
                                    rebase_to_margin: bool = False) -> dict:
    """
    Physical-fidelity importer:
      - Converts geometry to **meters** via _resolve_canvas_and_scale / _resolve_scale.
      - Resizes scene canvas to the physical canvas size (AUTO_CANVAS_POLICY='resize_scene').
      - No anisotropic scaling of geometry.
      - Positions can snap to GRID_CELL; sizes are NOT snapped by default.
      - Mirrors all placed objects to STATE for Blender livesync.
    """
    desired_w_m, desired_h_m, m_per_unit, is_px_like, src = _resolve_canvas_and_scale(plan)

    # ---- Live state: scene header ----
    STATE["grid_w"] = float(desired_w_m)
    STATE["grid_h"] = float(desired_h_m)
    STATE["objects"].clear()

    # 3) Start commands
    cmds = [{"tool": "reset_scene", "arguments": {}}]

    # 4) Resize scene to physical canvas (capped)
    if AUTO_CANVAS_POLICY == "resize_scene":
        Gw = min(desired_w_m, MAX_GRID_W)
        Gh = min(desired_h_m, MAX_GRID_H)
        cmds.append({"tool": "resize_canvas",
                     "arguments": {"grid_w": float(Gw), "grid_h": float(Gh)}})
    else:
        Gw = float(SCENE.get("grid_w", 40))
        Gh = float(SCENE.get("grid_h", 30))

    # 5) No rebase — keep origin
    dx = dy = 0.0

    def _center_size_px_to_meters(bb_px):
        x, y, w, h = [float(v) for v in bb_px]
        x_m = x * m_per_unit
        y_m = y * m_per_unit
        eps = 1e-4
        w_m = max(w * m_per_unit, eps)
        h_m = max(h * m_per_unit, eps)
        cx = x_m + w_m * 0.5 + dx
        cy = y_m + h_m * 0.5 + dy
        return _snap_pos(cx), _snap_pos(cy), _snap_size(w_m), _snap_size(h_m)

    # 6) Collect objects
    structurals, items, details = _collect_objects_anywhere(plan)

    # Optional wall bands: only if it's truly a room (outer/inner rects)
    ow = next((e["node"] for e in structurals if (e["key"] == "outer_wall" and (e["node"].get("type") == "rect"))), None)
    ia = next((e["node"] for e in structurals if (e["key"] == "inner_area" and (e["node"].get("type") == "rect"))), None)
    USE_WALL_BANDS = _looks_like_room(ow, ia)

    # 7) Walls / room import
    if USE_WALL_BANDS:
        xO, yO, wO, hO = _bbox_from_rect(ow)
        xI, yI, wI, hI = _bbox_from_rect(ia)

        def add_band(lbl, x, y, w, h):
            cx, cy, W, H = _center_size_px_to_meters((x, y, w, h))
            height_m = 2.7
            cmds.append({"tool": "add_object", "arguments": {
                "label": lbl, "primitive": "cube", "x": cx, "y": cy, "w": W, "h": H, "height": height_m
            }})
            # Mirror to livesync state
            STATE["objects"][lbl] = {"primitive": "cube", "x": float(cx), "y": float(cy),
                                     "w": float(W), "h": float(H), "height": float(height_m)}

        add_band("WALL_TOP",   xO, yO,        wO, (yI - yO))
        add_band("WALL_BOT",   xO, yI + hI,   wO, (yO + hO) - (yI + hI))
        add_band("WALL_LEFT",  xO, yI,        (xI - xO), hI)
        add_band("WALL_RIGHT", xI + wI, yI,   (xO + wO) - (xI + wI), hI)
    else:
        # Import room rects as thin slabs (safe for gardens/site plans)
        def _add_room_rect(node, label):
            if not node:
                return
            bb = _bbox_from_rect(node)
            cx, cy, W, H = _center_size_px_to_meters(bb)
            height_m = 0.1
            cmds.append({"tool": "add_object", "arguments": {
                "label": label, "primitive": "cube", "x": cx, "y": cy, "w": W, "h": H, "height": height_m
            }})
            STATE["objects"][label] = {"primitive": "cube", "x": float(cx), "y": float(cy),
                                       "w": float(W), "h": float(H), "height": float(height_m)}
        if ow: _add_room_rect(ow, "ROOM_OUTER")
        if ia: _add_room_rect(ia, "ROOM_INNER")

    # 8) Room details (rects only)
    for e in structurals:
        o, k = e["node"], e["key"]
        if k in ("outer_wall", "inner_area"):
            continue
        t = (o.get("type") or "").lower()
        if t != "rect":
            continue
        bb = _bbox_from_generic(o)
        if not bb:
            continue
        cx, cy, W, H = _center_size_px_to_meters(bb)
        lbl = _label_from(e)
        height_m = 1.2
        cmds.append({"tool": "add_object", "arguments": {
            "label": lbl, "primitive": "cube", "x": cx, "y": cy, "w": W, "h": H, "height": height_m
        }})
        STATE["objects"][lbl] = {"primitive": "cube", "x": float(cx), "y": float(cy),
                                 "w": float(W), "h": float(H), "height": float(height_m)}

    # 9) Items (furniture etc.)
    seen = set()
    for e in items:
        o = e["node"]
        bb = _bbox_from_generic(o)
        if not bb:
            continue
        cx, cy, W, H = _center_size_px_to_meters(bb)
        label = _label_from(e)
        if label in seen:
            i = 2
            while f"{label}_{i}" in seen:
                i += 1
            label = f"{label}_{i}"
        seen.add(label)

        # heights: convert per object; interprets units (m/cm/mm/px)
        height_m = _apply_height_cap(_height_m_from_obj(o, m_per_unit, is_px_like, default=DEFAULT_H))

        cmds.append({"tool": "add_object", "arguments": {
            "label": label, "primitive": "cube", "x": cx, "y": cy, "w": W, "h": H, "height": float(height_m)
        }})
        STATE["objects"][label] = {"primitive": "cube", "x": float(cx), "y": float(cy),
                                   "w": float(W), "h": float(H), "height": float(height_m)}

    # 10) Finish (write watcher once)
    _write_watch()
    cmds.append({"tool": "render_svg", "arguments": {"view": "topdown"}})
    cmds.append({"tool": "export_state", "arguments": {}})

    return {"commands": cmds}

def import_any_topdown_json_and_build(plan_json_str: str, write_to_watch: bool = False, watch_path: str = r"C:\ATLAS\live_scene.json"):
    """
    One-call helper:
      - loads arbitrary plan JSON
      - converts to agent batch (rects/cubes, snapped 0.5 m)
      - routes through engine
      - optionally writes exported state to Blender watcher
    """
    plan = json.loads(plan_json_str)
    batch = any_topdown_json_to_agent_batch(plan)
    outs = route_and_execute(batch, natural="[universal import]")
    if write_to_watch and "json" in outs:
        import shutil
        os.makedirs(os.path.dirname(watch_path), exist_ok=True)
        shutil.copyfile(outs["json"], watch_path)
        print(f"[LiveSync] wrote: {watch_path}")
    return outs
# ==== END UNIVERSAL IMPORTER ==================================================
if __name__ == "__main__":
    print("ATLAS_FINAL_WITH_IMPORTER module ready. Run via server_fixed.py.")
