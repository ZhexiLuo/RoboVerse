"""Generate training report for batch DP training on LIBERO-10 tasks."""

import glob
import re
from datetime import datetime
from pathlib import Path


TASKS = [
    "libero.pick_alphabet_soup",
    "libero.pick_bbq_sauce",
    "libero.pick_butter",
    "libero.pick_chocolate_pudding",
    "libero.pick_cream_cheese",
    "libero.pick_milk",
    "libero.orange_juice",
    "libero.pick_salad_dressing",
    "libero.pick_tomato_sauce",
]


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


def find_latest_stats(task: str) -> Path | None:
    """Find the most recent final_stats.txt for a task (sorted by filename = timestamp)."""
    pattern = f"tmp/{task}/diffusion_policy/franka/*/final_stats.txt"
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def main():
    rows = []
    total_success_all = 0
    total_completed_all = 0

    for task in TASKS:
        stats_path = find_latest_stats(task)
        if stats_path is None:
            rows.append((task, "❌ 未找到结果", "-", "-", "-"))
            continue

        stats = parse_final_stats(stats_path)
        if not stats:
            rows.append((task, "⚠️ 解析失败", "-", "-", "-"))
            continue

        success = stats.get("total_success", 0)
        completed = stats.get("total_completed", 0)
        rate = stats.get("success_rate", 0.0)
        total_success_all += success
        total_completed_all += completed

        status = "✅" if rate >= 0.5 else ("⚠️" if rate >= 0.2 else "❌")
        rows.append((task, status, success, completed, f"{rate:.2%}"))

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Diffusion Policy LIBERO-10 训练报告",
        "",
        f"生成时间：{date_str}  |  模型：ddpm_dit_model  |  max_step=350  |  epochs=100",
        "",
        "## 任务成功率汇总",
        "",
        "| 任务 | 状态 | 成功次数 | 总次数 | 成功率 |",
        "|------|------|----------|--------|--------|",
    ]
    for task, status, success, completed, rate in rows:
        lines.append(f"| `{task}` | {status} | {success} | {completed} | {rate} |")

    if total_completed_all > 0:
        overall = total_success_all / total_completed_all
        lines += [
            "",
            f"**总体平均成功率：{overall:.2%}** ({total_success_all}/{total_completed_all})",
        ]

    lines += [
        "",
        "## 详细日志",
        "",
        "每个任务的训练+评估日志位于：`roboverse_learn/claude/log/dp_{task}.log`",
        "",
        "每个任务的评估视频位于：`tmp/{task}/diffusion_policy/franka/{ckpt}/`",
    ]

    report_dir = Path("roboverse_learn/claude/out")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"dp_report_{datetime.now().strftime('%Y-%m-%d')}.md"
    report_path.write_text("\n".join(lines))

    print(f"📊 报告已生成：{report_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
