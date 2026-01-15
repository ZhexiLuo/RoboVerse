"""GPT-generated task: Stack the ketchup, bbq_sauce, and salad_dressing on top of each other like a sandwich, with the salad_dressing on top."""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .gpt_base import GptBaseTask


@register_task("gpt.sauce_sandwich", "gpt:SauceSandwich")
class SauceSandwichTask(GptBaseTask):
    scenario = ScenarioCfg(
        objects=[
        RigidObjCfg(name="ketchup", physics=PhysicStateType.RIGIDBODY, usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/ketchup/usd/ketchup.usd"),
        RigidObjCfg(name="bbq_sauce", physics=PhysicStateType.RIGIDBODY, usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd"),
        RigidObjCfg(name="salad_dressing", physics=PhysicStateType.RIGIDBODY, usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/salad_dressing/usd/salad_dressing.usd"),
        RigidObjCfg(name="butter", physics=PhysicStateType.RIGIDBODY, usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/butter/usd/butter.usd"),
        RigidObjCfg(name="orange_juice", physics=PhysicStateType.RIGIDBODY, usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/orange_juice/usd/orange_juice.usd")
        ],
        robots=["franka"],
    )
    max_episode_steps = 250
    task_desc = "Stack the ketchup, bbq_sauce, and salad_dressing on top of each other like a sandwich, with the salad_dressing on top."
    traj_filepath = "roboverse_data/trajs/gpt/sauce_sandwich/franka_v2.pkl"
