# blender_livesync.py — drop this into Blender's Text Editor and "Run Script" once
import bpy, json, os, time, math

WATCH_PATH = r"C:\ATLAS\live_scene.json"   # <- change if you prefer a different path
CHECK_EVERY = 0.5                          # seconds
_last_mtime = None
_last_sig = None 
GRID_NAME = "RefGrid"   # put this near your CONFIG block
CREATE_RAMP_ARROW = False   # ignore ramp_arrow objects coming from JSON
# --- Label display settings ---
DRAW_LABELS          = True                # master switch  # don't label structural shells
MIN_SIDE_FOR_LABEL   = 0.60                # m; skip if min(w,d) smaller than this
LABEL_SCALE          = 0.25                # fraction of min(w,d); was 0.40
LABEL_ABS_MAX        = 0.50                # m; clamp to avoid huge text
SKIP_PREFIXES      = ("WALL_", "WINDOW", "DOOR_", "ROOM_", "GRID_", "Label_")
LABEL_Z_OFFSET     = 0.002 

def _fresh_scene():
    # Clean scene without reloading the whole file (keeps your add-ons)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    # Add basic camera & light (optional)
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add(location=(18,18,16))
    if "Sun" not in bpy.data.objects:
        bpy.ops.object.light_add(type='SUN', location=(-10,-10,10))
        bpy.context.active_object.data.energy = 3.0

def _add_cube(name, x=0, y=0, cz=0, w=1, d=1, h=1):
    """Add a cube using CENTER-Z placement (cz)."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, cz))
    ob = bpy.context.active_object
    ob.name = name
    ob.dimensions = (w, d, h)
    ob.location.z = cz
def _apply_rotation(ob, a):
    from math import radians, atan2, fabs
    from mathutils import Matrix, Vector, Quaternion

    EPS = 1e-7

    r = a.get("rot_deg") or [0.0, 0.0, 0.0]
    yaw_deg  = float(r[2])                 # keep yaw AS-IS to match SVG/engine
    meta = a.get("meta") or {}
    h0 = meta.get("Hstart")
    h1 = meta.get("Hend")

    # --- 1) Build yaw first (so we get post-yaw local axes) ---
    R_yaw  = Matrix.Rotation(radians(yaw_deg), 4, 'Z')
    R_yaw3 = R_yaw.to_3x3()
    vX = (R_yaw3 @ Vector((1.0, 0.0, 0.0))).normalized()  # +local X in world
    vY = (R_yaw3 @ Vector((0.0, 1.0, 0.0))).normalized()  # +local Y in world (tilt axis)

    # --- 2) Decide tilt magnitude from rise & length (ignore engine tilt sign) ---
    # length = object X dimension BEFORE rotation; height/thickness = Z
    Lx = max(EPS, float(ob.dimensions.x))
    if isinstance(h0, (int, float)) and isinstance(h1, (int, float)):
        rise = float(h1 - h0)
        tilt_abs_deg = 0.0 if fabs(rise) < EPS else (atan2(fabs(rise), Lx) * 180.0 / 3.141592653589793)
    else:
        # fallback: use engine magnitude if heights missing
        tilt_abs_deg = fabs(float(r[0]))

    # --- 3) Try positive tilt, see which end goes up; flip if it disagrees with (Hend-Hstart) ---
    # Build rotation with +tilt_abs first
    Q_tilt_pos = Quaternion(vY, radians(tilt_abs_deg))
    R_pos = Q_tilt_pos.to_matrix().to_4x4() @ R_yaw

    need_flip = False
    if isinstance(h0, (int, float)) and isinstance(h1, (int, float)) and fabs(h1 - h0) > EPS:
        M3 = R_pos.to_3x3()
        hx = 0.5 * float(ob.dimensions.x)
        tz = 0.5 * float(ob.dimensions.z)

        # top-face centers at each end (start = -X, end = +X)
        p_start_top = M3 @ Vector((-hx, 0.0, +tz))
        p_end_top   = M3 @ Vector((+hx, 0.0, +tz))
        dz = p_end_top.z - p_start_top.z        # >0 means end (+X) higher

        desired_sign = 1.0 if (h1 - h0) > 0 else -1.0
        actual_sign  = 1.0 if dz > 0 else (-1.0 if dz < 0 else 0.0)

        # If ambiguous or wrong, we’ll flip the tilt sign
        if actual_sign == 0.0 or actual_sign != desired_sign:
            need_flip = True

    # --- 4) Final rotation ---
    if need_flip:
        Q_tilt_neg = Quaternion(vY, -radians(tilt_abs_deg))
        R = Q_tilt_neg.to_matrix().to_4x4() @ R_yaw
    else:
        R = R_pos

    ob.rotation_mode = 'XYZ'
    ob.rotation_euler = R.to_euler('XYZ')


def _maybe_add_label_for_object(label, x, y, cz, w, d, H):
    """Create a flat text label on top of the object if it passes filters."""
    if not DRAW_LABELS:
        return
    try:
        # filter out structural shells / tiny objects
        pfxes = ()
        try:
            pfxes = tuple(str(p).upper() for p in SKIP_PREFIXES)
        except NameError:
            pfxes = ()

        if any(str(label).upper().startswith(pfx) for pfx in pfxes):
            return

        if min(float(w), float(d)) < float(MIN_SIDE_FOR_LABEL):
            return
        # ... continue with label creation ...
    except Exception as e:
        print("[LiveSync][Label] Error:", e)


        # try to find the object we just created
        ob = bpy.data.objects.get(label)
        if ob is None:
            # try common variants (in case your mesh creation renames)
            candidates = (
                str(label),
                str(label).upper(),
                str(label).replace(" ", "_"),
                str(label).upper().replace(" ", "_"),
            )
            for n in candidates:
                ob = bpy.data.objects.get(n)
                if ob: break
        if ob is None and bpy.context.selected_objects:
            ob = bpy.context.selected_objects[-1]

        if ob is None:
            print("[LiveSync][Label] could not find object for:", label)
            return

        # place the text flat on the top face
        top_z  = cz + H * 0.5
        size_m = max(0.05, min(LABEL_ABS_MAX, LABEL_SCALE * min(w, d)))

        _add_label_text(
            label, x, y, top_z,
            parent=ob,
            size_m=size_m,
            z_offset=LABEL_Z_OFFSET,
            topdown=True
        )
    except Exception as e:
        print("[LiveSync][Label] Error:", e)

def _add_plane(name, x=0, y=0, z=0, w=1, d=1):
    bpy.ops.mesh.primitive_plane_add(location=(x, y, z))
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (w/2.0, d/2.0, 1.0)
# --- grid helpers (viewport-visible, not rendered) ---
# ---------- CONFIG ----------
CELL = 0.5          # 0.5 m agent grid
MARGIN_CELLS = 10    # how many cells of padding around layout
MIN_GRID_SIZE = 20    # extra cells around the layout
GRID_Z = -0.01

def _add_ramp(name, x=0, y=0, cz=0, L=1.0, W=1.0, T=0.1, rot_deg=(0,0,0)):
    # No L/W swapping; no yaw tweaks. L is along A→B (local +X), W is across (local +Y).
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, cz))
    ob = bpy.context.active_object
    ob.name = name
    ob.dimensions = (float(L), float(W), float(T))   # set BEFORE rotation
    _apply_rotation(ob, {"rot_deg": rot_deg})        # yaw then tilt (about local Y)
    return ob






def _add_ramp_arrow(name, parent, shaft_len, shaft_w, T, direction_up=True):
    """Draw a simple arrow on the ramp’s top surface, in parent (ramp) local space."""
    # Parent anchor (local space)
    empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty)
    empty.parent = parent
    empty.matrix_parent_inverse = parent.matrix_world.inverted()
    empty.location = (0.0, 0.0, T*0.5 + 0.002)

    # Shaft (plane), local to empty
    bpy.ops.mesh.primitive_plane_add()
    shaft = bpy.context.active_object
    shaft.name = name + "_shaft"
    shaft.parent = empty
    shaft.matrix_parent_inverse = empty.matrix_world.inverted()
    shaft.location = (0.0, 0.0, 0.0)
    shaft.scale = (float(shaft_len)*0.5, float(shaft_w)*0.5, 1.0)

    # Head (triangle): start as plane, collapse to a tip
    bpy.ops.mesh.primitive_plane_add()
    head = bpy.context.active_object
    head.name = name + "_head"
    head.parent = empty
    head.matrix_parent_inverse = empty.matrix_world.inverted()
    head.location = (float(shaft_len)*0.45, 0.0, 0.0)
    head.scale = (float(shaft_w)*0.9, float(shaft_w)*0.9, 1.0)

    bpy.context.view_layer.objects.active = head
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.merge(type='CENTER')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Flip by 180° in local Z if pointing “downhill”
    if not direction_up:
        empty.rotation_euler[2] += math.pi

    return shaft, head


def to_m(v): 
    return v * CELL
def _center_z_from_spec(a: dict, H: float) -> float:
    """
    Resolve CENTER Z.
    Priority:
      1) z_offset  (explicit center Z) – if nonzero, honor it; if 0, treat as 'ground it'
      2) z_bottom  (bottom Z) + H/2
      3) z         (legacy bottom Z) + H/2
      4) default: ground_z + H/2   (ground_z=0 if omitted)
    """
    ground_z = float(a.get("ground_z", 0.0))  # optional per-object ground level

    if "z_offset" in a:
        cz = float(a.get("z_offset") or 0.0)
        # If agent sent a real value (typically H/2), keep it; if it's 0, lift to ground.
        return cz if abs(cz) > 1e-9 else (ground_z + H * 0.5)

    if a.get("z_bottom") is not None:
        return float(a["z_bottom"]) + H * 0.5

    if a.get("z") is not None:
        return float(a["z"]) + H * 0.5

    # default: sit on ground
    return ground_z + H * 0.5



# --- improved grid creator (keeps true CELL spacing and auto size)
def ensure_ref_grid(spec, name=GRID_NAME, step=CELL, z=GRID_Z):
    # Remove existing
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    # JSON gives grid_w/grid_h in METERS  → convert to cells
    gw_m = float(spec.get("grid_w", 40.0))
    gh_m = float(spec.get("grid_h", 30.0))
    gw_cells = int(round(gw_m / CELL))
    gh_cells = int(round(gh_m / CELL))

    gw = gw_cells * CELL   # exact multiple of CELL
    gh = gh_cells * CELL


    mesh = bpy.data.meshes.new(name+"_mesh")
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)

    verts, edges = [], []

    # Horizontal lines
    for i in range(gh_cells + 1):
        y = -gh/2 + i * CELL
        v0 = len(verts)
        verts += [(-gw/2, y, z), (gw/2, y, z)]
        edges.append((v0, v0+1))

    # Vertical lines
    for j in range(gw_cells + 1):
        x = -gw/2 + j * CELL
        v0 = len(verts)
        verts += [(x, -gh/2, z), (x, gh/2, z)]
        edges.append((v0, v0+1))

    mesh.from_pydata(verts, edges, [])
    mesh.update()
    ob.display_type = 'WIRE'
    ob.show_in_front = False
    ob.hide_render = True
    ob.select_set(False)
    ob.hide_select = True

    return ob




def _bounds(objects_dict):
    xs, ys, ws, hs = [], [], [], []
    for a in objects_dict.values():
        xs.append(float(a.get("x", 0.0)))
        ys.append(float(a.get("y", 0.0)))
        ws.append(float(a.get("w") or a.get("width") or a.get("size") or 1.0))
        hs.append(float(a.get("h") or a.get("depth") or a.get("size") or 1.0))
    return min(xs), max(xs), min(ys), max(ys), max(ws or [1.0]), max(hs or [1.0])

def _anchor_offset(objects_dict):
    # If you have a special center label (e.g., "X"), anchor to it; else use centroid
    if "X" in objects_dict:
        a = objects_dict["X"]
        return float(a.get("x", 0.0)), float(a.get("y", 0.0))
    minx,maxx,miny,maxy,_,_ = _bounds(objects_dict)
    return (minx+maxx)/2.0, (miny+maxy)/2.0

def _auto_grid_params(objects_dict, spec=None):
    """
    If spec has grid_w/grid_h (units), use that (units * CELL).
    Otherwise, size the grid from object bounds.
    Returns (grid_size_meters, grid_at_xy, anchor_xy_in_svg).
    """
    if isinstance(spec, dict) and "grid_w" in spec and "grid_h" in spec:
        # grid_w/h are METERS in JSON; anchor must be in CELLS because x,y are cells
        gw_m = float(spec["grid_w"])
        gh_m = float(spec["grid_h"])
        cx = (gw_m / CELL) * 0.5   # cells
        cy = (gh_m / CELL) * 0.5   # cells

        # size only used for logging; keep a reasonable square
        w_m = max(MIN_GRID_SIZE, gw_m)
        h_m = max(MIN_GRID_SIZE, gh_m)
        size = max(w_m, h_m) + 2 * MARGIN_CELLS * CELL
        return size, (0.0, 0.0), (cx, cy)


    # fallback: from bounds
    minx,maxx,miny,maxy,maxw,maxh = _bounds(objects_dict)
    span_x = (maxx - minx) + maxw
    span_y = (maxy - miny) + maxh
    span   = max(span_x, span_y) + 2 * MARGIN_CELLS * CELL
    cells  = max(2, int(round(span / CELL)))
    size   = max(MIN_GRID_SIZE, cells * CELL)
    cx, cy = _anchor_offset(objects_dict)
    return size, (0.0, 0.0), (cx, cy)
def recenter_to_origin():
    # Move everything so the collective bounds center sits at (0,0)
    import bpy
    obs = [o for o in bpy.data.objects 
           if o.type in {'MESH','EMPTY','LIGHT','CAMERA'} 
           and o.name != GRID_NAME]
    if not obs:
        return
    minx = min(o.location.x for o in obs); maxx = max(o.location.x for o in obs)
    miny = min(o.location.y for o in obs); maxy = max(o.location.y for o in obs)
    cx = (minx+maxx)/2.0; cy = (miny+maxy)/2.0
    for o in obs:
        o.location.x -= cx
        o.location.y -= cy

def _add_label_text(text, x, y, z_top, parent=None, size_m=0.5, z_offset=0.001, topdown=True):
    # Create text aligned to WORLD axes, no implicit view alignment
    bpy.ops.object.text_add(
        location=(x, y, z_top + z_offset),
        rotation=(0.0, 0.0, 0.0),   # world-aligned on creation
        align='WORLD'
    )
    t = bpy.context.active_object
    t.name = f"Label_{text}"
    t.data.body = str(text)
    t.data.align_x = 'CENTER'
    t.data.align_y = 'CENTER'
    t.data.size   = float(size_m)

    # Force a known rotation: flat on XY (i.e., “painted” on the top face)
    t.rotation_mode = 'XYZ'
    if topdown:
        # FLAT on XY: try 0 first; if you see letters upside-down in top view, change Z to math.pi
        t.rotation_euler = (0.0, 0.0, 0.0)   # flat, readable from +Y
        # t.rotation_euler = (0.0, 0.0, math.pi)  # flat, flipped 180° if needed

    t.data.extrude = 0.001
    t.data.bevel_depth = 0.0

    if parent is not None:
        t.parent = parent
        t.matrix_parent_inverse = parent.matrix_world.inverted()
    return t



def _apply_objects(objects: dict, spec=None):
    _fresh_scene()

    # 1) grid sized to layout, centered at world origin
    grid_size, grid_at, anchor = _auto_grid_params(objects, spec)
    ensure_ref_grid(spec, step=CELL)
    if isinstance(spec, dict):
        gw = spec.get("grid_w"); gh = spec.get("grid_h")
        print(f"[LiveSync] grid_w/h: {gw} {gh} → grid_size(m): {grid_size}")

    # ---- decide position units (meters vs. cells) ----
    gw_m = float(spec.get("grid_w", 0.0)) if isinstance(spec, dict) else 0.0
    gh_m = float(spec.get("grid_h", 0.0)) if isinstance(spec, dict) else 0.0
    xs = [float(o.get("x", 0.0)) for o in objects.values()]
    ys = [float(o.get("y", 0.0)) for o in objects.values()]
    pos_are_meters = (gw_m > 0 and gh_m > 0 and max(xs + [0.0]) <= gw_m + 1e-6 and max(ys + [0.0]) <= gh_m + 1e-6)

    POS_SCALE = 1.0 if pos_are_meters else CELL
    if pos_are_meters:
        ox = gw_m * 0.5
        oy = gh_m * 0.5
    else:
        ox = (gw_m / CELL) * 0.5  # ok if grid_w not provided; becomes 0
        oy = (gh_m / CELL) * 0.5

    # Snap in meters (unless spec disables it)
    def snap(v_m):
        if isinstance(spec, dict) and spec.get("snap", True) is False:
            return v_m
        return round(v_m / CELL) * CELL

    # -------------------- FIRST PASS: create geometry --------------------
    for label, a in objects.items():
        prim = (a.get("primitive") or "cube").lower()

        # positions (SVG Y down → Blender Y up)
        x_in = float(a.get("x", 0.0))
        y_in = float(a.get("y", 0.0))
        x = snap((x_in - ox) * POS_SCALE)
        y = snap(-(y_in - oy) * POS_SCALE)

        # sizes are meters by default
        if not isinstance(spec, dict) or spec.get("size_units", "meters") == "meters":
            w = float(a.get("w") or a.get("width") or a.get("size") or 1.0)
            d = float(a.get("h") or a.get("depth") or a.get("size") or 1.0)
            H = float(a.get("height") or a.get("size") or 1.0)
        else:
            w = float(a.get("w") or 1.0) * CELL
            d = float(a.get("h") or 1.0) * CELL
            H = float(a.get("height") or 1.0) * CELL

        # vertical placement: center-Z convention
        cz = _center_z_from_spec(a, H)

        if prim == "plane":
            _add_plane(label, x, y, cz, w, d)

        elif prim == "ramp":
            rot_deg = a.get("rot_deg") or [0, 0, 0]
            _add_ramp(label, x, y, cz, L=w, W=d, T=H, rot_deg=rot_deg)

        elif prim == "ramp_arrow":
            # defer to second pass once parent exists
            continue

        else:
            # Create the cube (your function may or may not return the object; that's fine)
            _add_cube(label, x, y, cz, w, d, H)

            # Add label (kept inside the same scope where 'label' exists!)
            _maybe_add_label_for_object(label, x, y, cz, w, d, H)



    # -------------------- SECOND PASS: add ramp arrows --------------------
# -------------------- SECOND PASS: add ramp arrows --------------------
    if CREATE_RAMP_ARROW:
        for label, a in objects.items():
            if (a.get("primitive") or "").lower() != "ramp_arrow":
                continue
            parent = bpy.data.objects.get(a.get("parent"))
            if not parent:
                continue
            scl = a.get("scale") or [1.0, 0.2, 0.1]
            shaft_len, shaft_w, T = float(scl[0]), float(scl[1]), float(scl[2] or 0.1)
            direction_up = (a.get("dir", "up") == "up")
            _add_ramp_arrow(label, parent, shaft_len, shaft_w, T, direction_up=direction_up)



def _apply_commands(spec: dict):
    """If you prefer 'commands' batch (add_object/move/etc.), handle a minimal subset."""
    # For now: if 'objects' present, prefer that. Otherwise handle add_object-only.
    if "objects" in spec:
        _apply_objects(spec["objects"], spec)
        return
    _fresh_scene()
    for c in spec.get("commands", []):
        if c.get("tool") == "add_object":
            a = c.get("arguments", {}) or {}
            label = a.get("label") or "Obj"
            prim  = (a.get("primitive") or "cube").lower()
            x = float(a.get("x", 0)); y = float(a.get("y", 0))
            w = float(a.get("w") or a.get("width") or a.get("size") or 1)
            d = float(a.get("h") or a.get("depth") or a.get("size") or 1)
            H = float(a.get("height") or a.get("size") or 1)

            cz = _center_z_from_spec(a, H)

            if prim == "plane": _add_plane(label, x, y, cz, w, d)
            else:               _add_cube(label, x, y, cz, w, d, H)


def _tick():
    global _last_mtime, _last_sig
    try:
        if os.path.exists(WATCH_PATH):
            mtime = os.path.getmtime(WATCH_PATH)
            size = os.path.getsize(WATCH_PATH)
            with open(WATCH_PATH, "rb") as fb:
                head = fb.read(64)
                try:
                    fb.seek(max(0, size - 64))
                except Exception:
                    pass
                tail = fb.read(64)
            sig = (mtime, size, head, tail)

            # Run only if content or mtime changed
            if _last_sig != sig:
                _last_mtime = mtime
                _last_sig = sig
                with open(WATCH_PATH, "r", encoding="utf-8") as f:
                    spec = json.load(f)
                src = spec.get("_source", "?")
                print(f"[LiveSync] Detected change (src={src}); applying {WATCH_PATH}")
                _apply_commands(spec)
    except Exception as e:
        print("[LiveSync] Error:", e)
    return CHECK_EVERY


# Start timer (runs forever until you close Blender or unregister timers)
bpy.app.timers.register(_tick, first_interval=CHECK_EVERY, persistent=True)
print(f"[LiveSync] Watching: {WATCH_PATH}")

# Z semantics:
# - Engine writes z_offset (CENTER Z). This script uses center-origin placement by default.
# - Fallbacks supported: z_bottom + H/2, or legacy z + H/2 if z_offset absent.

import bpy

def enable_livesync():
    global _last_mtime, _last_sig
    _last_mtime = None
    _last_sig = None
    print(f"[LiveSync] ENABLED. Watching: {WATCH_PATH}")
    # avoid duplicate timers
    try:
        bpy.app.timers.unregister(_tick)
    except Exception:
        pass
    bpy.app.timers.register(_tick, first_interval=0.2, persistent=True)

def disable_livesync():
    try:
        bpy.app.timers.unregister(_tick)
        print("[LiveSync] DISABLED.")
    except Exception:
        pass

# auto-start when script is run
enable_livesync()
