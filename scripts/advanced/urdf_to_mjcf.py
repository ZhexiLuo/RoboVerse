#!/usr/bin/env python3
"""Convert URDF files to MJCF format using urdf2mjcf library."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from colorama import Fore, Style
from urdf2mjcf.convert import convert_urdf_to_mjcf

RESCALE_DIR = Path("metasim/cfg/tasks/rescale")


def postprocess_mjcf(mjcf_path: Path) -> None:
    """Remove incompatible attributes for dm_control compatibility."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    for geom in root.findall(".//geom"):
        if "scale" in geom.attrib:
            del geom.attrib["scale"]

    for body in root.findall(".//body"):
        for joint in body.findall("freejoint"):
            body.remove(joint)

    tree.write(mjcf_path, encoding="unicode", xml_declaration=True)


def apply_scale_json(mjcf_path: Path, scale_json_path: Path) -> None:
    """Apply scale, mass, and inertia from scale.json to MJCF."""
    with open(scale_json_path) as f:
        data = json.load(f)

    scale = data.get("scale", 1.0)
    mass = data.get("mass", 0.1)
    diaginertia = data.get("diaginertia", [0.001, 0.001, 0.001])

    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Apply scale to all mesh elements
    for mesh in root.findall(".//mesh"):
        mesh.set("scale", f"{scale} {scale} {scale}")

    # Add inertial to body
    body = root.find(".//body")
    if body is not None and body.find("inertial") is None:
        inertial = ET.Element("inertial")
        inertial.set("pos", "0 0 0")
        inertial.set("mass", str(mass))
        inertial.set("diaginertia", " ".join(f"{v:.6f}" for v in diaginertia))
        body.insert(0, inertial)

    tree.write(mjcf_path, encoding="unicode", xml_declaration=True)


def convert_asset(asset_dir: Path) -> bool:
    """Convert single asset URDF to MJCF."""
    urdf_path = asset_dir / "sample.urdf"
    mjcf_path = asset_dir / "sample.xml"
    scale_json = asset_dir / "scale.json"

    if not urdf_path.exists():
        print(Fore.RED + f"   ❌ Skip {asset_dir.name}: no sample.urdf" + Style.RESET_ALL)
        return False

    convert_urdf_to_mjcf(urdf_path, mjcf_path)
    postprocess_mjcf(mjcf_path)

    if scale_json.exists():
        apply_scale_json(mjcf_path, scale_json)
        print(Fore.GREEN + f"   ✅ {asset_dir.name} (with scale + inertial)" + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + f"   ⚠️  {asset_dir.name} (no scale.json)" + Style.RESET_ALL)

    return True


def main():
    """Convert all URDF files in rescale directory."""
    print(Fore.WHITE + Style.BRIGHT + "\n🔧 URDF to MJCF Conversion\n" + Style.RESET_ALL)

    asset_dirs = [d for d in RESCALE_DIR.iterdir() if d.is_dir() and d.name != "config"]

    for asset_dir in sorted(asset_dirs):
        try:
            convert_asset(asset_dir)
        except Exception as e:
            print(Fore.RED + f"   ❌ Failed {asset_dir.name}: {e}" + Style.RESET_ALL)

    print(Fore.WHITE + Style.BRIGHT + "\n🎉 Done!\n" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
