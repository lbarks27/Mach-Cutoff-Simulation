# Blender import helper for Mach Cutoff exports.
from __future__ import annotations

import json
from pathlib import Path

import bpy
import mathutils

BUNDLE_DIR = Path(__file__).resolve().parent


def _import_obj(path: Path):
    if not path.exists():
        return []
    for obj in list(bpy.context.selected_objects):
        obj.select_set(False)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    elif hasattr(bpy.ops.import_scene, "obj"):
        bpy.ops.import_scene.obj(filepath=str(path))
    else:
        raise RuntimeError("No OBJ importer found in this Blender build")
    return list(bpy.context.selected_objects)


def _ensure_material(name: str, rgba: tuple[float, float, float, float]):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    node_tree = material.node_tree
    if node_tree is None:
        return material
    bsdf = node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        return material
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.35
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = rgba
        bsdf.inputs["Emission Strength"].default_value = 0.0
    elif "Emission" in bsdf.inputs:
        bsdf.inputs["Emission"].default_value = rgba
    return material


def _assign_material(objects, material):
    for obj in objects:
        if obj.type not in {"MESH", "CURVE"}:
            continue
        data = obj.data
        if data is None:
            continue
        if len(data.materials) == 0:
            data.materials.append(material)
        else:
            data.materials[0] = material


def _frame_bounds(objects):
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    found = False
    for obj in objects:
        for corner in obj.bound_box:
            found = True
            world = obj.matrix_world @ mathutils.Vector(corner)
            mins[0] = min(mins[0], world.x)
            mins[1] = min(mins[1], world.y)
            mins[2] = min(mins[2], world.z)
            maxs[0] = max(maxs[0], world.x)
            maxs[1] = max(maxs[1], world.y)
            maxs[2] = max(maxs[2], world.z)
    if not found:
        return None, None
    return mins, maxs


def _ensure_camera_and_light(objects):
    mins, maxs = _frame_bounds(objects)
    if mins is None or maxs is None:
        return
    center = (
        0.5 * (mins[0] + maxs[0]),
        0.5 * (mins[1] + maxs[1]),
        0.5 * (mins[2] + maxs[2]),
    )
    span = max(maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2], 1.0)

    camera = bpy.data.objects.get("Camera")
    if camera is None:
        cam_data = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", cam_data)
        bpy.context.scene.collection.objects.link(camera)
    camera.location = (center[0] - 1.8 * span, center[1] - 1.4 * span, center[2] + 0.95 * span)
    direction = mathutils.Vector(center) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera

    sun = bpy.data.objects.get("Sun")
    if sun is None:
        light_data = bpy.data.lights.new(name="Sun", type="SUN")
        sun = bpy.data.objects.new(name="Sun", object_data=light_data)
        bpy.context.scene.collection.objects.link(sun)
    sun.location = (center[0] + span, center[1] - span, center[2] + 1.5 * span)
    sun.rotation_euler = (0.85, 0.2, 0.4)
    if sun.data is not None:
        sun.data.energy = 2.2


def main():
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.length_unit = "METERS"

    imported = []
    flight = _import_obj(BUNDLE_DIR / "flight_path.obj")
    rays = _import_obj(BUNDLE_DIR / "shock_rays.obj")
    hits = _import_obj(BUNDLE_DIR / "ground_hits.obj")
    terrain = _import_obj(BUNDLE_DIR / "terrain.obj")
    imported.extend(flight + rays + hits + terrain)

    _assign_material(terrain, _ensure_material("Terrain", (0.17, 0.22, 0.19, 1.0)))
    _assign_material(flight, _ensure_material("FlightPath", (0.92, 0.92, 0.92, 1.0)))
    _assign_material(rays, _ensure_material("ShockRays", (0.19, 0.45, 0.95, 1.0)))
    _assign_material(hits, _ensure_material("GroundHits", (0.82, 0.16, 0.17, 1.0)))

    _ensure_camera_and_light(imported)
    metadata_path = BUNDLE_DIR / "scene_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        bpy.context.scene["mach_cutoff_origin_lat_deg"] = metadata["origin_geodetic"]["lat_deg"]
        bpy.context.scene["mach_cutoff_origin_lon_deg"] = metadata["origin_geodetic"]["lon_deg"]
        bpy.context.scene["mach_cutoff_origin_alt_m"] = metadata["origin_geodetic"]["alt_m"]
    print(f"[mach_cutoff] imported Blender bundle from {BUNDLE_DIR}")


if __name__ == "__main__":
    main()
