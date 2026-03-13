from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__).resolve().parents[2]))

from gymnasium import make_vec
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver

from openpi_client import image_tools, websocket_client_policy


class PiPolicyRunner:
    """Queries a running π policy server and applies EE-delta actions via IK."""

    def __init__(
        self,
        env,
        scenario,
        num_envs: int,
        robot_name: str,
        policy_host: str,
        policy_port: int,
        image_size: int = 224,
        device: str = "cuda",
        actions_per_call: Optional[int] = None,
    ):
        if num_envs != 1:
            raise ValueError("pi_eval currently supports num_envs == 1")

        self.env = env
        self.scenario = scenario
        self.robot_name = robot_name
        self.image_size = image_size
        self.device = device
        self.actions_per_call = actions_per_call if actions_per_call and actions_per_call > 0 else None

        self.robot_cfg = scenario.robots[0]
        self.client = websocket_client_policy.WebsocketClientPolicy(host=policy_host, port=policy_port)

        # IK setup
        self.ik_solver = setup_ik_solver(self.robot_cfg, "pyroki")
        self.ee_body_name = self.robot_cfg.ee_body_name
        self.ee_body_idx: Optional[int] = None
        self.inverse_reorder_idx: Optional[list] = None

        self.cached_actions: Optional[np.ndarray] = None
        self.cache_index: int = 0
        self.cache_remaining: int = 0

    # ------------------------------------------------------------------
    def _ensure_reindex(self, rs) -> None:
        if self.inverse_reorder_idx is not None:
            return
        reorder_idx = self.env.task_env.handler.get_joint_reindex(self.robot_name)
        self.inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
        self.ee_body_idx = rs.body_names.index(self.ee_body_name)

    @staticmethod
    def _quat_wxyz_to_scipy(q: torch.Tensor) -> np.ndarray:
        """(B,4) wxyz -> scipy xyzw."""
        q = q.cpu().numpy()
        return np.concatenate([q[:, 1:], q[:, :1]], axis=-1)

    @staticmethod
    def _quat_scipy_to_wxyz(q: np.ndarray, device: str) -> torch.Tensor:
        """scipy xyzw (B,4) -> wxyz tensor."""
        return torch.from_numpy(np.concatenate([q[:, 3:], q[:, :3]], axis=-1)).to(device).float()

    def _build_ee_state(self, obs) -> np.ndarray:
        """Build 8-dim EE state: xyz(3) + axis_angle(3) + gripper(2) in world frame.

        LIBERO training data uses world-frame EE pos/quat (robot0_eef_pos, robot0_eef_quat),
        so we must also send world-frame state to the pi0_libero server.
        """
        rs = obs.robots[self.robot_name]
        self._ensure_reindex(rs)

        body_state = (rs.body_state if isinstance(rs.body_state, torch.Tensor) else torch.tensor(rs.body_state)).to(self.device).float()

        ee_p_world = body_state[:, self.ee_body_idx, :3]   # (1,3) world frame
        ee_q_world = body_state[:, self.ee_body_idx, 3:7]  # (1,4) wxyz world frame

        ee_rot_world = Rotation.from_quat(self._quat_wxyz_to_scipy(ee_q_world))
        ee_aa_world = ee_rot_world.as_rotvec()  # (1,3) axis-angle in world frame

        # Gripper: use last 2 joints (finger joints) as proxy for gripper opening
        joint_pos = rs.joint_pos.to(self.device).float()
        gripper_q = joint_pos[:, self.inverse_reorder_idx][:, 7:9].cpu().numpy()  # (1,2)

        state_8d = np.concatenate([ee_p_world[0].cpu().numpy(), ee_aa_world[0], gripper_q[0]]).astype(np.float32)
        return state_8d  # shape (8,)

    def _compress_image(self, obs) -> np.ndarray:
        cam = next(iter(obs.cameras.values()))
        rgb = cam.rgb
        rgb_np = rgb[0].detach().cpu().numpy() if rgb.dim() == 4 else rgb.detach().cpu().numpy()
        # LIBERO training images use [::-1, ::-1] flip (MuJoCo renders upside-down)
        rgb_np = np.ascontiguousarray(rgb_np[::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_np, self.image_size, self.image_size))

    def _get_prompt(self) -> str:
        task_env = getattr(self.env, "task_env", None)
        if task_env is not None and getattr(task_env, "task_desc", None):
            return str(task_env.task_desc)
        return "Execute the RoboVerse task."

    def _request_action_chunk(self, obs) -> None:
        img = self._compress_image(obs)
        state = self._build_ee_state(obs)
        policy_obs = {
            "observation/image": img,
            "observation/wrist_image": np.zeros_like(img),
            "observation/state": state,
            "prompt": self._get_prompt(),
        }
        response = self.client.infer(policy_obs)
        chunk = np.asarray(response["actions"], dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"Expected action chunk ndim=2, got {chunk.shape}")
        self.cached_actions = chunk
        self.cache_index = 0
        total = len(chunk)
        self.cache_remaining = total if self.actions_per_call is None else min(self.actions_per_call, total)

    def _decode_ee_action(self, obs, action_vec: np.ndarray) -> list[dict]:
        """Decode absolute EE target (xyz + axis_angle in world frame) -> IK -> joint targets.

        pi0_libero server applies AbsoluteActions after Unnormalize, so output is
        absolute EE target in world frame (same as LIBERO training data).
        pyroki IK expects target in local robot base frame.
        """
        rs = obs.robots[self.robot_name]
        self._ensure_reindex(rs)

        joint_pos_alpha = (rs.joint_pos if isinstance(rs.joint_pos, torch.Tensor) else torch.tensor(rs.joint_pos)).to(self.device).float()
        curr_q = joint_pos_alpha[:, self.inverse_reorder_idx]  # dict order (1, ndof)

        # action_vec[:3]  = absolute target EE position in world frame
        # action_vec[3:6] = absolute target EE orientation as axis-angle in world frame
        # action_vec[6]   = gripper command
        target_p_world = torch.tensor(action_vec[:3], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,3)
        target_rot_world = Rotation.from_rotvec(action_vec[3:6])

        # Convert world frame target to local robot base frame for pyroki IK
        root_state = (rs.root_state if isinstance(rs.root_state, torch.Tensor) else torch.tensor(rs.root_state)).to(self.device).float()
        robot_pos = root_state[:, :3]    # (1,3)
        robot_quat = root_state[:, 3:7]  # (1,4) wxyz

        inv_base_rot = Rotation.from_quat(self._quat_wxyz_to_scipy(robot_quat)).inv()
        target_p_local = torch.from_numpy(inv_base_rot.apply((target_p_world - robot_pos).cpu().numpy())).to(self.device).float()  # (1,3)
        target_rot_local = inv_base_rot * target_rot_world
        # target_rot_local may be a batch Rotation (1,); squeeze to single (4,) then add batch dim
        target_q_np = target_rot_local.as_quat().reshape(1, 4)  # (1,4) xyzw
        target_q = self._quat_scipy_to_wxyz(target_q_np, self.device)  # (1,4) wxyz

        # pi0_libero gripper: unnorm value in [-1, 1]; -1=close, 1=open
        # Convert to [0, 1] range for process_gripper_command (which uses > 0.5 threshold)
        gripper_normalized = torch.tensor([(action_vec[6] + 1.0) / 2.0], device=self.device)  # (1,)

        # IK in local frame
        q_sol, ik_ok = self.ik_solver.solve_ik_batch(target_p_local, target_q, curr_q)
        if not ik_ok.all():
            q_sol = curr_q  # fall back to current pose

        gripper_widths = process_gripper_command(gripper_normalized, self.robot_cfg, self.device)
        return self.ik_solver.compose_joint_action(q_sol, gripper_widths, current_q=curr_q, return_dict=True)

    def infer_action(self, obs) -> list[dict]:
        if (
            self.cached_actions is None
            or self.cache_remaining <= 0
            or self.cache_index >= len(self.cached_actions)
        ):
            self._request_action_chunk(obs)

        action_vec = self.cached_actions[self.cache_index]
        self.cache_index += 1
        self.cache_remaining -= 1
        return self._decode_ee_action(obs, action_vec)

    def reset(self) -> None:
        self.cached_actions = None
        self.cache_index = 0
        self.cache_remaining = 0

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def evaluate_episode(env, runner: PiPolicyRunner, max_steps: int, episode: int, output_dir: str) -> Dict[str, Any]:
    obs, info = env.reset()
    runner.reset()

    stats: Dict[str, Any] = {"steps": 0, "success": False, "total_reward": 0.0}

    os.makedirs(output_dir, exist_ok=True)
    obs_saver = ObsSaver(video_path=f"{output_dir}/episode_{episode:03d}.mp4")
    obs_saver.add(obs)

    for _ in range(max_steps):
        actions = runner.infer_action(obs)
        obs, reward, terminated, truncated, info = env.step(actions)

        stats["steps"] += 1
        stats["total_reward"] += float(reward.mean().item())
        obs_saver.add(obs)

        term = terminated.any().item() if hasattr(terminated, "any") else bool(terminated)
        trunc = truncated.any().item() if hasattr(truncated, "any") else bool(truncated)
        if term:
            stats["success"] = True
        if term or trunc:
            break

    obs_saver.save()
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate π policies via websocket server")
    parser.add_argument("--task", type=str, default="pick_butter")
    parser.add_argument("--robot", type=str, default="franka")
    parser.add_argument("--sim", type=str, default="mujoco")
    parser.add_argument("--policy-host", type=str, default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=350)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./pi_eval_output")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--actions-per-call", type=int, default=0,
                        help="0 = consume entire chunk; N = replan every N steps")
    # legacy arg kept for backward compat, unused
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--gripper-threshold", type=float, default=0.02)
    return parser.parse_args()


def main() -> bool:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_vec(
        f"RoboVerse/{args.task}",
        num_envs=args.num_envs,
        robots=[args.robot],
        simulator=args.sim,
        headless=True,
        cameras=[PinholeCameraCfg(
            name="camera",
            data_types=["rgb"],
            width=256,
            height=256,
            pos=(1.0, 0.0, 0.75),
            look_at=(0.0, 0.0, 0.0),
        )],
        device=args.device,
    )

    runner = PiPolicyRunner(
        env=env,
        scenario=env.scenario,
        num_envs=args.num_envs,
        robot_name=args.robot,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        image_size=args.image_size,
        device=args.device,
        actions_per_call=args.actions_per_call,
    )

    start_time = time.time()
    aggregate = {"total_episodes": 0, "total_successes": 0, "total_rewards": [], "episode_results": []}

    for ep in range(args.num_episodes):
        print(f"Episode {ep + 1}/{args.num_episodes}")
        result = evaluate_episode(env, runner, args.max_steps, ep + 1, args.output_dir)
        aggregate["total_episodes"] += 1
        aggregate["episode_results"].append(result)
        aggregate["total_rewards"].append(result["total_reward"])
        if result["success"]:
            aggregate["total_successes"] += 1
        sr = aggregate["total_successes"] / aggregate["total_episodes"]
        print(f"  Success rate so far: {sr:.1%}")

    total_time = time.time() - start_time
    final_sr = aggregate["total_successes"] / max(1, aggregate["total_episodes"])
    final_reward = float(np.mean(aggregate["total_rewards"])) if aggregate["total_rewards"] else 0.0
    print(f"\nEvaluation finished: success rate {final_sr:.1%}, avg reward {final_reward:.2f}, elapsed {total_time:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.output_dir, f"pi_eval_{args.task}_{ts}.json")
    with open(report_path, "w") as f:
        json.dump({"config": vars(args), "stats": aggregate, "timestamp": ts}, f, indent=2)
    print(f"Saved results to {report_path}")

    try:
        env.close()
    except Exception:
        pass
    runner.close()
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
