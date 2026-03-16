# RoboVerse VLA Training Pipeline

See the unified pipeline doc: [`../README.md`](../README.md)

## Supported Models

- **OpenVLA** (`OpenVLA/`) — 7B VLA, RLDS data format, LoRA fine-tuning
- **π₀ family** (`pi0/`) — π₀/π₀.₅/π₀-FAST, LeRobot data format, LoRA fine-tuning
- **SmolVLA** (`SmolVLA/`) — Lightweight VLA from HuggingFace/LeRobot

## Key Files

| File | Description |
|------|-------------|
| `OpenVLA/finetune.sh` | OpenVLA LoRA fine-tuning |
| `OpenVLA/vla_eval.py` | OpenVLA evaluation |
| `OpenVLA/setup_env.sh` | Environment setup (rlds_env + openvla conda) |
| `pi0/train_pi0.sh` | π₀ one-click training |
| `pi0/pi_eval.py` | π₀ evaluation client |
| `pi0/convert_roboverse_to_lerobot.py` | RoboVerse → LeRobot converter |
| `rlds_utils/roboverse/roboverse.py` | RoboVerse → RLDS converter |
