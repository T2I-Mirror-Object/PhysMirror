#!/usr/bin/env python3
"""
debug_scene_colored.py – Text → Colored 3D Mirror Scene (GLB)

Runs Stage 1 (text → mesh) then assembles the full mirror scene in trimesh so
that original mesh colors and UV textures are preserved throughout.  The existing
Stage 2 pipeline paints all meshes white before rendering; this script bypasses
that step and works entirely in trimesh to keep object colours intact.

Output: a single .glb file containing
  • coloured objects (one mesh per extracted object)
  • mirror-reflected copies of every object
  • mirror frame (dark grey)
  • mirror glass (light blue, semi-transparent)
  • floor  (optional)
  • walls  (optional)

All layout parameters are read from configs/inference.yaml (Stage 2 section).

Usage
-----
  python tests/debug_scene_colored.py \\
      --prompt "a wooden chair, a red vase in front of the mirror, in a tidy room"

  python tests/debug_scene_colored.py \\
      --prompt "a toaster, a bottle, a cup in front of the mirror, in a kitchenette" \\
      --force --no_walls
"""

import sys
import os
import argparse
import random
import math

import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stage1_mesh.objects_extractors import get_objects_extractor
from src.stage1_mesh.factory import get_text_to_3d_model


# ─────────────────────────────────────────────────────────────────────────────
# Trimesh geometry helpers  (all preserve vertex colours / UV textures)
# ─────────────────────────────────────────────────────────────────────────────

def load_colored_mesh(path: str) -> trimesh.Trimesh:
    """Load OBJ / GLB preserving vertex colours and UV textures."""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if not meshes:
            raise ValueError(f"No geometry found in {path}")
        # Single sub-mesh → return directly (preserves TextureVisuals)
        if len(meshes) == 1:
            return meshes[0]
        # Multiple sub-meshes → concatenate (may collapse multi-material UV)
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"Unexpected type {type(loaded)} loading {path}")


def ground_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Translate so the lowest vertex sits exactly at Y = 0."""
    m = mesh.copy()
    m.apply_translation([0.0, -m.bounds[0, 1], 0.0])
    return m


def scale_around_centroid(mesh: trimesh.Trimesh, factor: float) -> trimesh.Trimesh:
    """Uniform scale about the mesh centroid."""
    m = mesh.copy()
    c = m.centroid.copy()
    m.apply_translation(-c)
    m.apply_scale(factor)
    m.apply_translation(c)
    return m


def rotate_y_around_centroid(mesh: trimesh.Trimesh, angle_rad: float) -> trimesh.Trimesh:
    """Rotate the mesh around its own centroid on the Y axis."""
    m = mesh.copy()
    c = m.centroid.copy()
    m.apply_translation(-c)
    m.apply_transform(tra.rotation_matrix(angle_rad, [0, 1, 0]))
    m.apply_translation(c)
    return m


def reflect_across_z_plane(mesh: trimesh.Trimesh, mirror_z: float) -> trimesh.Trimesh:
    """
    Mirror the mesh across the plane Z = mirror_z.

    Steps:
      1. Apply the reflection transform  Z' = -Z + 2·mirror_z
      2. Reverse face winding order so normals point toward the viewer.

    Vertex colours and UV textures live in texture-space and are unchanged.
    """
    m = mesh.copy()
    # Reflection + translation encoded as a single 4×4 matrix
    T = np.eye(4, dtype=np.float64)
    T[2, 2] = -1.0
    T[2, 3] = 2.0 * mirror_z
    m.apply_transform(T)
    # Flip winding so the reflected surface faces the right way
    m.faces = m.faces[:, ::-1]
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Scene component factories
# ─────────────────────────────────────────────────────────────────────────────

def _solid(tri_mesh: trimesh.Trimesh, rgba) -> trimesh.Trimesh:
    """Assign a uniform RGBA vertex colour to every vertex of tri_mesh."""
    tri_mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(rgba, (len(tri_mesh.vertices), 1)).astype(np.uint8)
    )
    return tri_mesh


def make_mirror_frame(w: float, h: float, thickness: float, depth: float = 0.1) -> trimesh.Trimesh:
    """Hollow rectangular frame via boolean difference."""
    outer = trimesh.creation.box([w, h, depth])
    inner = trimesh.creation.box([max(w - 2 * thickness, 0.01),
                                  max(h - 2 * thickness, 0.01),
                                  depth * 1.2])
    frame = outer.difference(inner)
    return _solid(frame, [80, 80, 80, 255])


def make_mirror_glass(inner_w: float, inner_h: float, depth: float = 0.005) -> trimesh.Trimesh:
    """Thin glass pane (light blue, semi-transparent)."""
    glass = trimesh.creation.box([inner_w, inner_h, depth])
    return _solid(glass, [200, 225, 255, 180])


def make_floor(room_w: float, room_d: float) -> trimesh.Trimesh:
    b = trimesh.creation.box([room_w, 0.1, room_d])
    b.apply_translation([0, -0.05, 0])   # top face at Y = 0
    return _solid(b, [210, 200, 190, 255])


def make_walls(room_w: float, room_d: float, wall_h: float = 10.0,
               thickness: float = 0.5) -> list:
    specs = [
        # extents                         translation
        ([room_w,    wall_h, thickness], [0,                        wall_h / 2, -room_d / 2 - thickness / 2]),  # back
        ([room_w,    wall_h, thickness], [0,                        wall_h / 2,  room_d / 2 + thickness / 2]),  # front
        ([thickness, wall_h, room_d],   [-room_w / 2 - thickness / 2, wall_h / 2, 0]),                          # left
        ([thickness, wall_h, room_d],   [ room_w / 2 + thickness / 2, wall_h / 2, 0]),                          # right
    ]
    walls = []
    for extents, trans in specs:
        b = trimesh.creation.box(extents)
        b.apply_translation(trans)
        walls.append(_solid(b, [220, 215, 205, 255]))
    return walls


# ─────────────────────────────────────────────────────────────────────────────
# Layout (mirrors Stage 2 LayoutEngine logic, in trimesh)
# ─────────────────────────────────────────────────────────────────────────────

def group_bounds(meshes) -> np.ndarray:
    """Axis-aligned bounding box of the whole group: shape (2, 3)."""
    mins = [m.bounds[0] for m in meshes]
    maxs = [m.bounds[1] for m in meshes]
    return np.array([np.min(mins, axis=0), np.max(maxs, axis=0)])


def arrange_objects(meshes: list, gap: float,
                    base_rotation_deg: float, random_rotation: bool) -> list:
    """
    Replicate Stage 2 LayoutEngine.arrange():
      Ground → base Y-rotation → random Y-rotation → side-by-side → centre at X=0.
    """
    # 1. Ground
    meshes = [ground_mesh(m) for m in meshes]

    # 2. Base Y-rotation (applied to every object)
    if base_rotation_deg != 0.0:
        rad = math.radians(base_rotation_deg)
        meshes = [rotate_y_around_centroid(m, rad) for m in meshes]

    # 3. Optional random Y-rotation
    if random_rotation:
        meshes = [rotate_y_around_centroid(m, random.uniform(-math.pi / 4, math.pi / 4))
                  for m in meshes]

    # 4. Place objects side-by-side along X with the given gap
    positioned = [meshes[0]]
    prev = meshes[0]
    for mesh in meshes[1:]:
        shift_x = prev.bounds[1, 0] - mesh.bounds[0, 0] + gap
        m = mesh.copy()
        m.apply_translation([shift_x, 0.0, 0.0])
        positioned.append(m)
        prev = m

    # 5. Centre the whole group at X = 0
    first_min_x = positioned[0].bounds[0, 0]
    last_max_x  = positioned[-1].bounds[1, 0]
    cx = (first_min_x + last_max_x) / 2.0
    centred = []
    for m in positioned:
        mc = m.copy()
        mc.apply_translation([-cx, 0.0, 0.0])
        centred.append(mc)

    return centred


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(args):
    print("=" * 60)
    print("DEBUG: Text → Coloured 3D Mirror Scene (GLB)")
    print("=" * 60)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg_path = args.config or os.path.join(
        os.path.dirname(__file__), "..", "configs", "inference.yaml")
    cfg = load_yaml(cfg_path)
    s1  = cfg.get("stage1", {})
    s2  = cfg.get("stage2", {})

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mesh_dir = os.path.join(args.output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    # ── Stage 1: Text → Mesh files ────────────────────────────────────────────
    print(f"\n[Stage 1] Prompt: '{args.prompt}'")

    extractor = get_objects_extractor(s1.get("extractor", "simple2"))
    descs = extractor.extract(args.prompt)
    print(f"[Stage 1] Extracted objects: {descs}")

    model_name = s1.get("mesh_model", "trellis")
    mesh_ext   = ".glb" if model_name == "trellis" else ".obj"

    mesh_kwargs: dict = {"device": device}
    if model_name == "trellis":
        mesh_kwargs["model_name"] = s1.get("trellis_model_name",
                                           "microsoft/TRELLIS-text-large")

    mesh_model = get_text_to_3d_model(model_name, **mesh_kwargs)

    mesh_paths = []
    for i, desc in enumerate(descs):
        safe = desc.replace(" ", "_").replace("/", "_")[:60]
        out  = os.path.join(mesh_dir, f"{i}_{safe}{mesh_ext}")
        if args.force and os.path.exists(out):
            os.remove(out)
        print(f"[Stage 1] Generating '{desc}'...")
        mesh_model.generate(desc, out)
        mesh_paths.append(out)

    del mesh_model
    torch.cuda.empty_cache()

    # ── Load meshes (colours preserved) ──────────────────────────────────────
    print("\n[Scene] Loading meshes with original colours...")
    raw: list = []
    for p in mesh_paths:
        m = load_colored_mesh(p)
        print(f"  {p}  —  {len(m.vertices):,} verts, "
              f"visual: {type(m.visual).__name__}")
        raw.append(m)

    # ── Scale ─────────────────────────────────────────────────────────────────
    obj_scale = s2.get("object_scale", 1.5)
    if obj_scale != 1.0:
        raw = [scale_around_centroid(m, obj_scale) for m in raw]
        print(f"[Layout] Scaled objects ×{obj_scale}")

    # ── Layout ────────────────────────────────────────────────────────────────
    objects = arrange_objects(
        raw,
        gap               = s2.get("gap", 0.5),
        base_rotation_deg = s2.get("object_base_rotation", 180.0),
        random_rotation   = s2.get("random_rotation", True),
    )

    # ── Auto-scale to fixed mirror_height (if configured) ─────────────────────
    mirror_height  = s2.get("mirror_height", None)
    mirror_gap_top = s2.get("mirror_gap_top", 2.0)
    if mirror_height is not None:
        gb      = group_bounds(objects)
        group_h = gb[1, 1] - gb[0, 1]
        avail_h = mirror_height - mirror_gap_top
        if group_h > 0 and avail_h > 0:
            sf      = avail_h / group_h
            objects = [scale_around_centroid(m, sf) for m in objects]
            objects = [ground_mesh(m) for m in objects]
            print(f"[Layout] Auto-scaled ×{sf:.3f} to fit mirror_height={mirror_height}")

    # ── Compute mirror dimensions ──────────────────────────────────────────────
    gb               = group_bounds(objects)
    group_w          = gb[1, 0] - gb[0, 0]
    group_h          = gb[1, 1] - gb[0, 1]
    mirror_gap_side  = s2.get("mirror_gap_side",  2.0)
    mirror_gap_ahead = s2.get("mirror_gap_ahead", 1.7)
    mirror_thickness = s2.get("mirror_thickness", 0.1)

    frame_w  = max(group_w + 2 * mirror_gap_side, 2.0)
    frame_h  = max(group_h + mirror_gap_top,       2.0)
    mirror_z = -mirror_gap_ahead

    print(f"[Scene] Objects group: {group_w:.2f} W × {group_h:.2f} H")
    print(f"[Scene] Mirror frame:  {frame_w:.2f} W × {frame_h:.2f} H  @ Z={mirror_z:.2f}")

    # ── Reflections ────────────────────────────────────────────────────────────
    print("[Scene] Creating coloured mirror reflections...")
    reflections = [reflect_across_z_plane(m, mirror_z) for m in objects]

    # ── Assemble trimesh Scene ────────────────────────────────────────────────
    scene = trimesh.scene.Scene()

    for i, m in enumerate(objects):
        scene.add_geometry(m, geom_name=f"object_{i}")
        print(f"  + object_{i}")

    for i, m in enumerate(reflections):
        scene.add_geometry(m, geom_name=f"reflection_{i}")
        print(f"  + reflection_{i}")

    # Mirror frame (box centered at origin → ground it → move to mirror_z)
    frame         = make_mirror_frame(frame_w, frame_h, mirror_thickness)
    frame_y_shift = -frame.bounds[0, 1]          # lift bottom face to Y = 0
    frame.apply_translation([0.0, frame_y_shift, mirror_z])
    scene.add_geometry(frame, geom_name="mirror_frame")
    print(f"  + mirror_frame")

    # Mirror glass (same centre as frame, sits just behind the frame plane)
    inner_w = frame_w - 2 * mirror_thickness
    inner_h = frame_h - 2 * mirror_thickness
    glass   = make_mirror_glass(inner_w, inner_h)
    glass.apply_translation([0.0, frame_y_shift, mirror_z])
    scene.add_geometry(glass, geom_name="mirror_glass")
    print(f"  + mirror_glass")

    if not args.no_floor:
        scene.add_geometry(make_floor(20.0, 20.0), geom_name="floor")
        print(f"  + floor")

    if not args.no_walls:
        for i, w in enumerate(make_walls(20.0, 20.0)):
            scene.add_geometry(w, geom_name=f"wall_{i}")
        print(f"  + wall_0..3")

    # ── Export GLB ────────────────────────────────────────────────────────────
    out_path = args.output or os.path.join(args.output_dir, "colored_scene.glb")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    scene.export(out_path)

    print(f"\n[Success] Scene written to: {out_path}")
    print(f"  Meshes: {list(scene.geometry.keys())}")
    print()
    print("  Open in Blender : File › Import › glTF 2.0 (*.glb/*.gltf)")
    print("  Online viewer   : https://gltf-viewer.donmccurdy.com")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text prompt → coloured 3D mirror scene exported as GLB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--prompt", "-p", type=str,
        default="a wooden chair in front of the mirror, in a modern room",
        help="Scene description (same format as the main pipeline).",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to inference.yaml. Defaults to configs/inference.yaml.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output .glb path. Default: <output_dir>/colored_scene.glb.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/debug_colored_scene",
        help="Working directory for mesh cache and output.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete cached meshes and regenerate from scratch.",
    )
    parser.add_argument(
        "--no_floor", action="store_true",
        help="Omit the floor plane.",
    )
    parser.add_argument(
        "--no_walls", action="store_true",
        help="Omit the four room walls.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device: cuda or cpu (auto-detected if not set).",
    )

    args = parser.parse_args()
    run(args)
