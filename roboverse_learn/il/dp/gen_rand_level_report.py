"""Compare Diffusion Policy success rates across randomization levels."""

import argparse
import glob
import re
from datetime import datetime
from pathlib import Path


LEVELS = [0, 1, 2, 3]
LEVEL_DESC = {
    0: "No randomization",
    1: "Physics only",
    2: "Physics + Materials",
    3: "Physics + Materials + Cameras",
}


def parse_final_stats(stats_path: Path) -> dict:
    """Parse final_stats.txt; returns keys: total_success, total_completed, success_rate."""
    result = {}
    text = stats_path.read_text()
    for line in text.splitlines():
        if m := re.match(r"Total Success: (\d+)", line):
            result["total_success"] = int(m.group(1))
        elif m := re.match(r"Total Completed: (\d+)", line):
            result["total_completed"] = int(m.group(1))
        elif m := re.match(r"Average Average Success Rate: ([0-9.]+)", line):
            result["success_rate"] = float(m.group(1))
    return result


def find_latest_stats(task_l: str) -> Path | None:
    """Find the most recent final_stats.txt for a task-level."""
    pattern = f"tmp/{task_l}/diffusion_policy/franka/*/final_stats.txt"
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def main():
    parser = argparse.ArgumentParser(description="Generate rand-level ablation report.")
    parser.add_argument("--task", default="libero.pick_chocolate_pudding")
    args = parser.parse_args()

    rows = []
    for level in LEVELS:
        task_l = f"{args.task}-L{level}"
        stats_path = find_latest_stats(task_l)
        if stats_path is None:
            rows.append((level, LEVEL_DESC[level], "❌ not found", "-", "-", "-"))
            continue
        stats = parse_final_stats(stats_path)
        if not stats:
            rows.append((level, LEVEL_DESC[level], "⚠️ parse error", "-", "-", "-"))
            continue
        success = stats.get("total_success", 0)
        completed = stats.get("total_completed", 0)
        rate = stats.get("success_rate", 0.0)
        status = "✅" if rate >= 0.5 else ("⚠️" if rate >= 0.2 else "❌")
        rows.append((level, LEVEL_DESC[level], status, success, completed, f"{rate:.2%}"))

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Randomization Level Ablation Report — {args.task}",
        "",
        f"生成时间：{date_str}  |  模型：ddpm_dit_model  |  epochs=100  |  max_step=350",
        "",
        "## 各 Level 成功率对比",
        "",
        "| Level | 配置 | 状态 | 成功 | 总计 | 成功率 |",
        "|-------|------|------|------|------|--------|",
    ]
    for level, desc, status, success, completed, rate in rows:
        lines.append(f"| L{level} | {desc} | {status} | {success} | {completed} | {rate} |")

    lines += [
        "",
        "## 日志路径",
        "",
        *[f"- L{lvl}: `claude/log/rand_level_collect_L{lvl}.log` + `roboverse_learn/claude/log/dp_{args.task}-L{lvl}.log`" for lvl in LEVELS],
    ]

    report_dir = Path("roboverse_learn/claude/out")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"rand_level_report_{datetime.now().strftime('%Y-%m-%d')}.md"
    report_path.write_text("\n".join(lines))

    print(f"📊 报告已生成：{report_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
