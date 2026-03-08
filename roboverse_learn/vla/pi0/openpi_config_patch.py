"""
Script to inject RoboVerse config into third_party/openpi/src/openpi/training/config.py.
Run from RoboVerse root AFTER installing openpi.

Usage:
    cd third_party/openpi && uv sync && cd ../..
    python roboverse_learn/vla/pi0/openpi_config_patch.py
"""
import sys
from pathlib import Path

ROBOVERSE_ROOT = Path(__file__).parents[3]
CONFIG_FILE = ROBOVERSE_ROOT / "third_party/openpi/src/openpi/training/config.py"
POLICY_SRC = ROBOVERSE_ROOT / "roboverse_learn/vla/pi0/roboverse_policy.py"
POLICY_DEST = ROBOVERSE_ROOT / "third_party/openpi/src/openpi/policies/roboverse_policy.py"

IMPORT_PATCH = "import openpi.policies.roboverse_policy as roboverse_policy"
IMPORT_ANCHOR = "import openpi.policies.libero_policy as libero_policy"

DATA_CONFIG_CLASS = '''

@dataclasses.dataclass(frozen=True)
class _LeRobotLiberoDataConfig_RoboVerse(DataConfigFactory):
    """Data config for RoboVerse demos: 9-dim joint_pos state, roboverse_policy transforms."""

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[roboverse_policy.RoboVerseInputs(model_type=model_config.model_type)],
            outputs=[roboverse_policy.RoboVerseOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(-2, 7)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
'''

TRAIN_CONFIG = '''
    # RoboVerse: pi0 with LoRA for joint_pos control (9-dim state: 7 arm + 2 gripper)
    TrainConfig(
        name="pi0_roboverse_lora",
        model=pi0_config.Pi0Config(
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=_LeRobotLiberoDataConfig_RoboVerse(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
    ),
'''

RLDS_CLASS_ANCHOR = "\n@dataclasses.dataclass(frozen=True)\nclass RLDSDroidDataConfig(DataConfigFactory):"
CONFIGS_END_ANCHOR = "    *roboarena_config.get_roboarena_configs(),\n    *polaris_config.get_polaris_configs(),\n]"


def main():
    if not CONFIG_FILE.exists():
        print(f"❌ openpi config not found: {CONFIG_FILE}")
        print("   Run: cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync")
        sys.exit(1)

    # Copy roboverse_policy
    POLICY_DEST.write_text(POLICY_SRC.read_text())
    print(f"✅ Copied roboverse_policy.py → {POLICY_DEST}")

    content = CONFIG_FILE.read_text()

    if "roboverse_policy" in content:
        print("ℹ️  roboverse_policy already registered in config.py")
        return

    # Inject import
    content = content.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + "\n" + IMPORT_PATCH)

    # Inject data config class before RLDSDroidDataConfig
    content = content.replace(
        RLDS_CLASS_ANCHOR,
        DATA_CONFIG_CLASS + RLDS_CLASS_ANCHOR,
    )

    # Inject TrainConfig before closing ]
    content = content.replace(
        CONFIGS_END_ANCHOR,
        CONFIGS_END_ANCHOR.replace("]", TRAIN_CONFIG + "]"),
    )

    CONFIG_FILE.write_text(content)
    print(f"✅ Injected pi0_roboverse_lora config into {CONFIG_FILE}")

    # Verify
    import subprocess
    result = subprocess.run(
        ["python", "-c", "from openpi.training.config import get_config; get_config('pi0_roboverse_lora'); print('✅ verified')"],
        capture_output=True, text=True,
        cwd=str(ROBOVERSE_ROOT / "third_party/openpi"),
        env={"PATH": "/usr/bin:/bin"},
    )
    print(result.stdout or result.stderr)


if __name__ == "__main__":
    main()
