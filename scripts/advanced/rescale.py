#!/usr/bin/env python3
"""Estimate object scale and mass using LLM, output to gpt_gen compatible format."""

import json
import os
import time
from pathlib import Path

import numpy as np
import openai
import trimesh
from colorama import Fore, Style

# Configuration
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://yunwu.ai/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")

RESCALE_DIR = Path("metasim/cfg/tasks/rescale")
DESCRIPTIONS_FILE = RESCALE_DIR / "descriptions.json"
OUTPUT_JSON = RESCALE_DIR / "config" / "rigid_objects_init_list.json"


def load_descriptions() -> dict:
    """Load object descriptions from JSON file."""
    with open(DESCRIPTIONS_FILE) as f:
        return json.load(f)


def compute_bbox(mesh_path: Path) -> dict:
    """Compute bounding box dimensions and z_min from OBJ mesh."""
    mesh = trimesh.load(mesh_path)
    bounds = mesh.bounds
    dims = bounds[1] - bounds[0]
    return {"x": float(dims[0]), "y": float(dims[1]), "z": float(dims[2]), "z_min": float(bounds[0][2])}


def call_llm(description: str, bbox: dict) -> dict:
    """Call LLM to estimate real-world dimensions and mass."""
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""You are an expert at estimating real-world object properties.

Object description: {description}
Current mesh bounding box (arbitrary units): x={bbox['x']:.4f}, y={bbox['y']:.4f}, z={bbox['z']:.4f}

Estimate the real-world dimensions (in METERS) and mass (in KG) of this object.
Output strictly valid JSON with no extra text:
{{"real_x": <float>, "real_y": <float>, "real_z": <float>, "mass_kg": <float>}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1].replace("json", "").strip()
    return json.loads(content)


def compute_scale(bbox: dict, real_dims: dict) -> float:
    """Compute uniform scale factor."""
    scales = [
        real_dims["real_x"] / bbox["x"] if bbox["x"] > 0 else 1.0,
        real_dims["real_y"] / bbox["y"] if bbox["y"] > 0 else 1.0,
        real_dims["real_z"] / bbox["z"] if bbox["z"] > 0 else 1.0,
    ]
    return float(np.mean(scales))


def compute_z_offset(z_min: float, scale: float) -> float:
    """Compute z offset to place object bottom on table surface with 50cm drop height."""
    return -z_min * scale + 0.5


def compute_diaginertia(mass: float, real_dims: dict) -> list:
    """Compute diagonal inertia tensor (approximating as solid box)."""
    x, y, z = real_dims["real_x"], real_dims["real_y"], real_dims["real_z"]
    ixx = (1 / 12) * mass * (y**2 + z**2)
    iyy = (1 / 12) * mass * (x**2 + z**2)
    izz = (1 / 12) * mass * (x**2 + y**2)
    return [float(ixx), float(iyy), float(izz)]


def save_scale_json(asset_dir: Path, data: dict) -> Path:
    """Save scale data to JSON file."""
    path = asset_dir / "scale.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def process_asset(name: str, description: str) -> dict | None:
    """Process single asset, return entry for rigid_objects_init_list."""
    asset_dir = RESCALE_DIR / name
    mesh_path = asset_dir / "mesh" / "sample.obj"

    if not mesh_path.exists():
        print(Fore.RED + f"   ❌ Skip {name}: mesh not found" + Style.RESET_ALL)
        return None

    print(Fore.CYAN + f"\n🔍 Processing: {name}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"   📝 Description: {description}" + Style.RESET_ALL)

    bbox = compute_bbox(mesh_path)
    print(Fore.BLUE + f"   📦 Mesh bbox: x={bbox['x']:.3f}, y={bbox['y']:.3f}, z={bbox['z']:.3f}" + Style.RESET_ALL)

    print(Fore.MAGENTA + "   🤖 Calling LLM..." + Style.RESET_ALL)
    llm_result = call_llm(description, bbox)
    print(Fore.GREEN + "   🤖 LLM Response:" + Style.RESET_ALL)
    print(f"      real_dims: {llm_result['real_x']:.3f}m x {llm_result['real_y']:.3f}m x {llm_result['real_z']:.3f}m")
    print(f"      mass: {llm_result['mass_kg']:.3f} kg")

    # Compute derived values
    scale = compute_scale(bbox, llm_result)
    z_offset = compute_z_offset(bbox["z_min"], scale)
    diaginertia = compute_diaginertia(llm_result["mass_kg"], llm_result)

    print(Fore.CYAN + "   📐 Computed:" + Style.RESET_ALL)
    print(f"      scale: {scale:.4f}")
    print(f"      z_offset: {z_offset:.4f}")
    print(f"      diaginertia: [{diaginertia[0]:.6f}, {diaginertia[1]:.6f}, {diaginertia[2]:.6f}]")

    # Save scale.json
    scale_data = {
        "scale": scale,
        "mass": llm_result["mass_kg"],
        "diaginertia": diaginertia,
        "z_offset": z_offset,
        "mesh_bbox": bbox,
        "real_dimensions": {
            "x": llm_result["real_x"],
            "y": llm_result["real_y"],
            "z": llm_result["real_z"],
        },
    }
    save_scale_json(asset_dir, scale_data)
    print(Fore.GREEN + "   ✅ Saved: scale.json" + Style.RESET_ALL)

    # Return entry for rigid_objects_init_list
    return {
        "init_state": {
            "pos": ["@x", "@y", z_offset],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        "filepath": str(asset_dir / "sample.xml"),
    }


def main():
    """Process all assets and generate output JSON."""
    print(Fore.WHITE + Style.BRIGHT + "\n🚀 Rescale Pipeline Started\n" + Style.RESET_ALL)

    descriptions = load_descriptions()
    total = len(descriptions)
    output_data = {}
    task_times = []

    start_time = time.time()

    for i, (name, info) in enumerate(descriptions.items(), 1):
        task_start = time.time()
        print(Fore.WHITE + f"[{i}/{total}] " + Style.RESET_ALL, end="")

        result = process_asset(name, info["description"])

        task_elapsed = time.time() - task_start
        task_times.append(task_elapsed)

        if result:
            output_data[name] = result
            print(Fore.CYAN + f"   ⏱️  {task_elapsed:.2f}s" + Style.RESET_ALL)

    # Save rigid_objects_init_list.json
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)

    # Statistics
    total_time = time.time() - start_time
    success = len(output_data)
    avg_time = np.mean(task_times) if task_times else 0

    print(Fore.WHITE + Style.BRIGHT + f"\n🎉 Done! Output: {OUTPUT_JSON}" + Style.RESET_ALL)
    print(Fore.GREEN + f"   ✅ Success: {success}/{total} ({100*success/total:.1f}%)" + Style.RESET_ALL)
    print(Fore.BLUE + f"   ⏱️  Total time: {total_time:.2f}s" + Style.RESET_ALL)
    print(Fore.YELLOW + f"   📊 Avg per task: {avg_time:.2f}s" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
