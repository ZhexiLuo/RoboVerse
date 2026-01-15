#!/usr/bin/env python3

import json
import os
import pickle
import re
from pathlib import Path

import emoji
import openai
from colorama import Fore, Style

# ======================================
# 1. Configuration & Setup
# ======================================

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://yunwu.ai/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Centralized path configuration
CONFIG_DIR = Path("metasim/cfg/tasks/gpt/config")
OBJECT_LIST_JSON = CONFIG_DIR / "rigid_objects_init_list.json"
ROBOT_LIST_JSON = CONFIG_DIR / "robots_init_list.json"
TASKS_OUTPUT_FOLDER = CONFIG_DIR / "tasks"
PKL_OUTPUT_BASE = Path("roboverse_data/trajs/gpt")
TASK_OUTPUT_FOLDER = Path("roboverse_pack/tasks/gpt")


# -------------- JSON LOADING UTILS --------------


def load_all_objects_data():
    """Load the entire objects_init_list.json and return as a dict."""
    if not OBJECT_LIST_JSON.is_file():
        raise FileNotFoundError(f"Cannot find {OBJECT_LIST_JSON}")
    with open(OBJECT_LIST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_all_robots_data():
    """Load the entire robots_init_list.json and return as a dict."""
    if not ROBOT_LIST_JSON.is_file():
        raise FileNotFoundError(f"Cannot find {ROBOT_LIST_JSON}")
    with open(ROBOT_LIST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_available_objects():
    """Return list of object names (keys) from object_init_list.json."""
    return list(load_all_objects_data().keys())


def load_available_robots():
    """Return list of robot names (keys) from robot_init_list.json."""
    return list(load_all_robots_data().keys())


# ======================================
# Utility: Conversions of naming
# ======================================


def strip_markdown_code_block(content: str) -> str:
    """Remove markdown code block wrapper if present."""
    if content.startswith("```"):
        content = content[3:-3] if content.endswith("```") else content[3:]
        content = content.replace("json", "").strip()
    return content


def to_snake_case(s: str) -> str:
    """Convert a string to snake_case. E.g., 'Sauce Pyramid' -> 'sauce_pyramid'."""
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    return "_".join(s.lower().split())


def to_camel_case(s: str) -> str:
    """Convert a string to CamelCase. E.g., 'Sauce Pyramid' -> 'SaucePyramid'."""
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    return "".join(word.capitalize() for word in s.split())


def call_gpt(system_prompt: str, user_prompt: str = "") -> dict:
    """Call GPT and return parsed JSON."""
    messages = [{"role": "system", "content": system_prompt}]
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=0.7, max_tokens=15000
    )
    content = strip_markdown_code_block(response.choices[0].message.content.strip())
    return json.loads(content)


# ======================================
# 2. First GPT Call (Partial Task JSON)
# ======================================


def call_gpt_to_generate_task(user_prompt, object_list, robot_list):
    """
    GPT must output strictly valid JSON:
      {
        "task_name": "...",
        "task_language_instruction": "...",
        "robot_involved": [...],
        "objects_involved": [...]
      }
    """
    system_instructions = (
        "You are a helpful assistant that creates tabletop-manipulation tasks based on user requests.\n"
        "We have the following objects:\n"
        f"{object_list}\n\n"
        "We have the following robots:\n"
        f"{robot_list}\n\n"
        "You must:\n"
        "1) Pick exactly one or more robots from the above.\n"
        "2) Pick zero or more objects from the above.\n"
        "3) Invent a short, unique task name.\n"
        "4) Write a one or two sentence 'task_language_instruction' describing an unusual or amusing task.\n"
        "   - Example_1: 'Put the basket upside down to cover the orange juice like a hat.'\n"
        "   - Example_2: 'Place butter, chocolate pudding and milk in a line and knock them over like dominoes.'\n"
        "   - Keep it short, no more than 2 sentences.\n"
        "5) Output strictly valid JSON **only** in the following format:\n"
        "{\n"
        '  "task_name": "...",\n'
        '  "task_language_instruction": "...",\n'
        '  "robot_involved": [...],\n'
        '  "objects_involved": [...]\n'
        "}\n\n"
        "Constraints:\n"
        "- No extraneous keys.\n"
        "- Do not wrap in triple backticks.\n"
        "- The 'task_name' must be unique.\n"
        "- The 'task_language_instruction' must be 1-2 sentences at most.\n"
    )

    return call_gpt(system_instructions, user_prompt)


# ======================================
# 3. Second GPT Call: Append "init_state" to Partial JSON
# ======================================


def call_gpt_to_get_init_state(partial_task_json, all_objects_data, all_robots_data):
    """
    GPT merges:
      - The chosen robot(s) (with known pos, rot, dof_pos from all_robots_data).
      - The chosen objects (plus any decorative objects, if GPT desires).
        For each object: x,y in [-0.5, 0.5], z and rot from the library.

    GPT outputs strictly valid JSON:
    {
      "init_state": {
        "robotName": {
          "pos": [...],
          "rot": [...],
          "dof_pos": {...}
        },
        "object1": {
          "pos": [...],
          "rot": [...]
        },
        ...
      }
    }
    """
    task_name = partial_task_json["task_name"]
    robot_involved = partial_task_json["robot_involved"]
    objects_involved = partial_task_json["objects_involved"]
    task_instr = partial_task_json["task_language_instruction"]

    # Condense objects
    condensed_objs = {}
    for name, data in all_objects_data.items():
        condensed_objs[name] = {
            "z": data["init_state"]["pos"][2],
            "rot": data["init_state"]["rot"],
            "filepath": data["filepath"],
        }

    # Condense robots
    condensed_robs = {}
    for rname, rdata in all_robots_data.items():
        condensed_robs[rname] = {"pos": rdata["pos"], "rot": rdata["rot"], "dof_pos": rdata["dof_pos"]}

    # GPT Prompt
    system_instructions = (
        "You are a helpful assistant that finalizes the 'init_state' for a RoboVerse-like scene.\n"
        "We have a partial task specification:\n"
        f'  task_name: "{task_name}"\n'
        f'  task_language_instruction: "{task_instr}"\n'
        f"  robot_involved: {robot_involved}\n"
        f"  objects_involved: {objects_involved}\n\n"
        "We have a library of all possible objects with fixed z & rot:\n"
        f"{json.dumps(condensed_objs, indent=2)}\n\n"
        "We have these possible robots:\n"
        f"{json.dumps(condensed_robs, indent=2)}\n\n"
        "Your job:\n"
        "1) Possibly add more 'decorative' objects (any from the object library) if you want.\n"
        "2) For each object (required + decorative), pick x,y in [-0.5, 0.5].\n"
        "   Then combine that with the known z,rot from the library.\n"
        "3) For each chosen robot, just copy its pos,rot,dof_pos from the library exactly.\n"
        "4) Output strictly valid JSON with exactly one top-level key:\n"
        '   "init_state" : { ... }.\n'
        '5) Inside "init_state", each key is either a robot name or object name.\n'
        '   The value is { "pos": [x,y,z], "rot": [w,x,y,z], (optionally "dof_pos" if it is the robot) }.\n'
        "6) No extra keys.\n"
        "7) Do NOT wrap in triple backticks.\n"
        '8) No actions, no states, no extra in this JSON—only the "init_state" dict.\n'
    )

    return call_gpt(system_instructions)


# ======================================
# 4. Write Task .py
# ======================================
def write_task_file(final_json, object_library):
    """Generate a task file in roboverse_pack/tasks/gpt/{snake_task_name}.py."""
    task_name = final_json["task_name"]
    snake_task_name = to_snake_case(task_name)
    camel_task_name = to_camel_case(task_name)
    task_class_name = camel_task_name + "Task"

    out_py = TASK_OUTPUT_FOLDER / f"{snake_task_name}.py"

    robots_involved = final_json.get("robot_involved", [])
    init_state_data = final_json["init_state"]
    task_desc = final_json.get("task_language_instruction", "")

    # Build objects list
    object_list_entries = []
    for name in init_state_data.keys():
        if name in robots_involved:
            continue
        if name not in object_library:
            print(f"Warning: object '{name}' not found in library; skipping.")
            continue
        filepath = object_library[name]["filepath"]
        entry = f'        RigidObjCfg(name="{name}", physics=PhysicStateType.RIGIDBODY, usd_path="{filepath}")'
        object_list_entries.append(entry)

    objects_str = ",\n".join(object_list_entries) if object_list_entries else ""
    traj_path = PKL_OUTPUT_BASE / snake_task_name / "franka_v2.pkl"

    py_content = f'''"""GPT-generated task: {task_desc}"""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.scenario.objects import RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from .gpt_base import GptBaseTask


@register_task("gpt.{snake_task_name}", "gpt:{camel_task_name}")
class {task_class_name}(GptBaseTask):
    scenario = ScenarioCfg(
        objects=[
{objects_str}
        ],
        robots=["franka"],
    )
    max_episode_steps = 250
    task_desc = "{task_desc}"
    traj_filepath = "{traj_path}"
'''

    TASK_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    with open(out_py, "w", encoding="utf-8") as f:
        f.write(py_content)

    return str(out_py), snake_task_name, task_class_name, camel_task_name


# ======================================
# 5. Task Generation
# ======================================


def get_user_prompt() -> str:
    """Get user input prompt."""
    print(Fore.YELLOW + emoji.emojize("🔥 What can I help you with today? ✨") + Style.RESET_ALL)
    prompt = input("> ").strip()
    return prompt or "Please generate an interesting task for me."


def generate_task(user_prompt: str, all_objs: dict, all_robs: dict) -> dict:
    """Generate task via two GPT calls."""
    partial = call_gpt_to_generate_task(user_prompt, list(all_objs.keys()), list(all_robs.keys()))
    for key in ["task_name", "task_language_instruction", "robot_involved", "objects_involved"]:
        if key not in partial:
            raise ValueError(f"GPT missing key '{key}' in partial_task.")
    init_state = call_gpt_to_get_init_state(partial, all_objs, all_robs)
    if "init_state" not in init_state:
        raise ValueError("GPT did not provide 'init_state' top-level in second call.")
    partial["init_state"] = init_state["init_state"]
    return partial


# ======================================
# 6. Writers
# ======================================


def write_task_json(task: dict) -> str:
    """Write task JSON file."""
    snake_name = to_snake_case(task["task_name"])
    path = TASKS_OUTPUT_FOLDER / f"{snake_name}.json"
    TASKS_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)
    return str(path)


def write_task_pkl(task: dict) -> str:
    """Write task PKL file."""
    snake_name = to_snake_case(task["task_name"])
    roverse_data = {}
    for robot_name in task["robot_involved"]:
        if robot_name not in task["init_state"]:
            raise ValueError(f"Robot '{robot_name}' not found in final init_state JSON.")
        dof_pos_dict = task["init_state"][robot_name].get("dof_pos", {})
        zero_dof_target = {joint: 0.0 for joint in dof_pos_dict.keys()}
        roverse_data[robot_name] = [
            {"actions": [{"dof_pos_target": zero_dof_target}], "init_state": task["init_state"], "states": [], "extra": None}
        ]
    folder = PKL_OUTPUT_BASE / snake_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "franka_v2.pkl"
    with open(path, "wb") as f:
        pickle.dump(roverse_data, f)
    return str(path)


# ======================================
# 7. UI
# ======================================


def print_summary(task: dict, json_path: str, pkl_path: str, cfg_path: str, snake_name: str, cfg_class: str, camel_name: str):
    """Print colored summary."""
    print("\n" + Fore.GREEN + emoji.emojize("🚀 The task has been generated! 🎉") + Style.RESET_ALL)
    print(Fore.CYAN + "🔹 Task Name: " + Style.BRIGHT + task["task_name"] + Style.RESET_ALL)
    print(Fore.MAGENTA + "📝 Task Language Instruction: " + Style.BRIGHT + task["task_language_instruction"] + Style.RESET_ALL + "\n")
    print(Fore.BLUE + "📁 Task JSON saved to:" + Style.RESET_ALL)
    print(Fore.YELLOW + f"  {json_path}" + Style.RESET_ALL)
    print(Fore.BLUE + "📦 Task PKL saved to:" + Style.RESET_ALL)
    print(Fore.YELLOW + f"  {pkl_path}" + Style.RESET_ALL)
    print(Fore.BLUE + "🔧 Task Python file saved to:" + Style.RESET_ALL)
    print(Fore.YELLOW + f"  {cfg_path}" + Style.RESET_ALL + "\n")
    print(Fore.GREEN + emoji.emojize("🎮 You can replay your task by running:") + Style.RESET_ALL)
    print(Fore.WHITE + f"  python scripts/advanced/replay_demo.py --sim=mujoco --task=gpt.{snake_name} --num_envs 1" + Style.RESET_ALL + "\n")


# ======================================
# 8. Main Workflow
# ======================================


def main():
    user_prompt = get_user_prompt()
    all_objs, all_robs = load_all_objects_data(), load_all_robots_data()
    task = generate_task(user_prompt, all_objs, all_robs)
    json_path = write_task_json(task)
    pkl_path = write_task_pkl(task)
    cfg_path, snake_name, cfg_class, camel_name = write_task_file(task, all_objs)
    print_summary(task, json_path, pkl_path, cfg_path, snake_name, cfg_class, camel_name)


if __name__ == "__main__":
    main()
