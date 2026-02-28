# Rescale Pipeline

Convert custom URDF assets to MuJoCo-compatible MJCF format with LLM-estimated physical properties.

## Overview

This pipeline processes URDF assets and prepares them for use with the `gpt_gen.py` task generation system.

```
rescale.py → urdf_to_mjcf.py → gpt_gen.py
```

## Workflow

### Step 1: Estimate Scale and Mass

```bash
python scripts/advanced/rescale.py
```

**Input:**
- `metasim/cfg/tasks/rescale/descriptions.json` - Object descriptions
- `metasim/cfg/tasks/rescale/{asset}/mesh/sample.obj` - Mesh files

**Output:**
- `metasim/cfg/tasks/rescale/{asset}/scale.json` - Physical properties
- `metasim/cfg/tasks/rescale/config/rigid_objects_init_list.json` - gpt_gen compatible format

**LLM estimates:**
- `scale` - Uniform scale factor to real-world size
- `mass` - Object mass in kg

**Computed from LLM output:**
- `z_offset` - Height offset for table placement: `bbox_z * scale / 2`
- `diaginertia` - Inertia tensor (solid box approximation)

### Step 2: Convert URDF to MJCF

```bash
python scripts/advanced/urdf_to_mjcf.py
```

**Input:**
- `metasim/cfg/tasks/rescale/{asset}/sample.urdf`
- `metasim/cfg/tasks/rescale/{asset}/scale.json`

**Output:**
- `metasim/cfg/tasks/rescale/{asset}/sample.xml` - MuJoCo MJCF

Uses [urdf2mjcf](https://github.com/kscalelabs/urdf2mjcf) library with post-processing to add inertial properties.

### Step 3: Generate Tasks (Optional)

```bash
python scripts/advanced/gpt_gen.py
```

Uses `config/rigid_objects_init_list.json` to generate manipulation tasks.


### step4 :visulize
run command according to step3 output, if you run in headless server, using `MUJOCO_GL=egl [python command ] --headless`
## Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional
export OPENAI_MODEL="gpt-4o-2024-08-06"             # Optional
```

## File Structure

```
metasim/cfg/tasks/rescale/
├── descriptions.json              # Object descriptions (input)
├── config/
│   └── rigid_objects_init_list.json  # gpt_gen input (output)
└── {asset}/
    ├── sample.urdf                # Original URDF
    ├── sample.xml                 # Converted MJCF
    ├── scale.json                 # Physical properties
    └── mesh/
        ├── sample.obj             # Visual mesh
        └── sample_collision.obj   # Collision mesh
```

## future development: Adding New Assets

1. Create asset directory: `metasim/cfg/tasks/rescale/{asset_name}/`
2. Add URDF file: `sample.urdf`
3. Add mesh files: `mesh/sample.obj`, `mesh/sample_collision.obj`
4. Add description to `descriptions.json`
5. Run pipeline: `rescale.py` → `urdf_to_mjcf.py`

## todo
using image input, remove rely on descriptions.json