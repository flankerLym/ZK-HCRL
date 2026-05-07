
# -*- coding: utf-8 -*-
"""
科研风格绘图脚本：解析 TCO-DRL 运行日志 txt，并输出可直接用于论文的中文图表

用法：
    python plot_tco_results_cn.py --input "26_5_7_18_05.txt"
    python plot_tco_results_cn.py --input "26_5_7_18_05.txt" --output_dir "./paper_figs"

输出：
    1) final_总奖励.png
    2) final_成功率.png
    3) final_平均响应时间.png
    4) final_平均成本.png
    5) final_恶意节点分配次数.png
    6) final_可信节点分配次数.png
    7) final_节点类型分配构成.png
    8) final_综合指标热力图.png
    9) trend_成功率训练趋势.png
   10) trend_总奖励训练趋势.png
   11) trend_DQN_PPO_RA对比.png
   12) summary_results.csv
   13) summary_normalized.csv

说明：
- 自动解析 “Final results” 区域和每个 Episode 内的指标
- 图表采用中文标题与坐标标签，适合论文或答辩直接使用
- 不依赖 seaborn，仅使用 matplotlib
"""

import argparse
import ast
import csv
import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_chinese_font():
    # 常见中文字体候选；不同机器可能只存在其中一部分
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
        'WenQuanYi Micro Hei', 'Arial Unicode MS', 'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 160
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_array_block(text, key):
    """
    解析类似：
    success_rate:
    [0.251 0.268 ...]
    """
    pattern = re.compile(rf"{re.escape(key)}:\s*\n(\[[\s\S]*?\])", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return None
    raw = m.group(1)
    # 将多空格转为逗号形式再解析
    raw_clean = raw.replace('\n', ' ')
    raw_clean = re.sub(r'\s+', ' ', raw_clean).strip()
    inner = raw_clean.strip()[1:-1].strip()
    if not inner:
        return []
    parts = inner.split(' ')
    vals = []
    for p in parts:
        p = p.strip().strip(',')
        if not p:
            continue
        try:
            vals.append(float(p))
        except:
            pass
    return vals


def parse_method_order(text):
    pattern = re.compile(r"method order:\s*\n(\[[\s\S]*?\])", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        return None
    return ast.literal_eval(m.group(1))


def parse_episode_metrics(text):
    """
    解析每个 episode 中：
    [Method] reward: ... avg_responseT: ... success_rate: ... success_time_rate: ... finishT: ... Cost: ...
    返回:
        episode_data[method] = {metric: [values...]}
    """
    episode_blocks = re.split(r"-{10,}Episode\s+\d+\s+-{10,}", text)
    # 第一个 split 前是 header，真正 episode block 从 1 开始
    episode_blocks = episode_blocks[1:]
    episode_data = {}
    for epi_idx, block in enumerate(episode_blocks):
        # 到下一段 "Episode" 或 "Final results" 前
        lines = block.splitlines()
        for line in lines:
            line = line.strip()
            if not line.startswith('['):
                continue
            # 例：
            # [Random] reward: 228.424  avg_responseT: 5.294 success_rate: 0.245 ...
            m = re.match(
                r"\[(.*?)\]\s+reward:\s*([-\d\.]+)\s+avg_responseT:\s*([-\d\.]+)\s+success_rate:\s*([-\d\.]+)\s+success_time_rate:\s*([-\d\.]+)\s+finishT:\s*([-\d\.]+)\s+Cost:\s*([-\d\.]+)",
                line
            )
            if not m:
                continue
            method = m.group(1)
            reward = float(m.group(2))
            avg_responseT = float(m.group(3))
            success_rate = float(m.group(4))
            success_time_rate = float(m.group(5))
            finishT = float(m.group(6))
            cost = float(m.group(7))

            if method not in episode_data:
                episode_data[method] = {
                    "episode": [],
                    "reward": [],
                    "avg_responseT": [],
                    "success_rate": [],
                    "success_time_rate": [],
                    "finishT": [],
                    "cost": [],
                }
            episode_data[method]["episode"].append(epi_idx)
            episode_data[method]["reward"].append(reward)
            episode_data[method]["avg_responseT"].append(avg_responseT)
            episode_data[method]["success_rate"].append(success_rate)
            episode_data[method]["success_time_rate"].append(success_time_rate)
            episode_data[method]["finishT"].append(finishT)
            episode_data[method]["cost"].append(cost)
    return episode_data


def build_final_dataframe(text):
    methods = parse_method_order(text)
    if methods is None:
        raise ValueError("未找到 Final results 中的 method order。")

    metrics_map = {
        "avg_responseT": "平均响应时间",
        "total_rewards": "总奖励",
        "success_rate": "成功率",
        "success_time_rate": "按时成功率",
        "finishT": "完成时间",
        "cost": "平均成本",
        "match rate": "类型匹配率",
        "requests assigned to malicious oracle": "分配到恶意节点次数",
        "requests assigned to normal oracle": "分配到普通节点次数",
        "requests assigned to trusted oracle": "分配到可信节点次数",
    }

    values = {}
    for eng in metrics_map:
        arr = parse_array_block(text, eng)
        if arr is None:
            raise ValueError(f"未找到指标: {eng}")
        values[eng] = arr

    df = pd.DataFrame({"方法": methods})
    for eng, cn in metrics_map.items():
        df[cn] = values[eng]
    return df


def save_dataframe(df, out_dir):
    df.to_csv(Path(out_dir) / "summary_results.csv", index=False, encoding="utf-8-sig")

    norm_df = df.copy()
    metric_cols = [c for c in norm_df.columns if c != "方法"]

    # 归一化：高优指标正向归一化，低优指标反向归一化
    bigger_better = {"总奖励", "成功率", "按时成功率", "类型匹配率", "分配到可信节点次数"}
    smaller_better = {"平均响应时间", "完成时间", "平均成本", "分配到恶意节点次数", "分配到普通节点次数"}

    out = pd.DataFrame({"方法": norm_df["方法"]})
    for col in metric_cols:
        x = norm_df[col].astype(float).values
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if abs(xmax - xmin) < 1e-12:
            norm = np.ones_like(x)
        else:
            if col in bigger_better:
                norm = (x - xmin) / (xmax - xmin)
            elif col in smaller_better:
                norm = (xmax - x) / (xmax - xmin)
            else:
                norm = (x - xmin) / (xmax - xmin)
        out[col] = norm

    out["综合得分"] = out.drop(columns=["方法"]).mean(axis=1)
    out.to_csv(Path(out_dir) / "summary_normalized.csv", index=False, encoding="utf-8-sig")
    return out


def _bar_plot(df, xcol, ycol, title, ylabel, out_path, annotate_fmt="{:.3f}"):
    plt.figure(figsize=(9, 5.3))
    bars = plt.bar(df[xcol], df[ycol])
    plt.title(title)
    plt.xlabel("方法")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha='right')
    ymax = max(df[ycol]) if len(df[ycol]) > 0 else 1
    ymin = min(df[ycol]) if len(df[ycol]) > 0 else 0
    pad = (ymax - ymin) * 0.08 if ymax != ymin else 0.1
    plt.ylim(ymin - pad * 0.2, ymax + pad)
    for bar, val in zip(bars, df[ycol]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + pad*0.1,
                 annotate_fmt.format(val), ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_stacked_assignment(df, out_path):
    plt.figure(figsize=(9.5, 5.8))
    x = np.arange(len(df))
    malicious = df["分配到恶意节点次数"].values
    normal = df["分配到普通节点次数"].values
    trusted = df["分配到可信节点次数"].values

    plt.bar(x, malicious, label="恶意节点")
    plt.bar(x, normal, bottom=malicious, label="普通节点")
    plt.bar(x, trusted, bottom=malicious+normal, label="可信节点")

    plt.xticks(x, df["方法"], rotation=25, ha='right')
    plt.ylabel("请求分配次数")
    plt.xlabel("方法")
    plt.title("各方法的节点类型分配构成")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_heatmap(norm_df, out_path):
    show_cols = [c for c in norm_df.columns if c != "方法"]
    data = norm_df[show_cols].values

    plt.figure(figsize=(10.5, 5.8))
    plt.imshow(data, aspect='auto')
    plt.colorbar(label="归一化得分")
    plt.yticks(np.arange(len(norm_df)), norm_df["方法"])
    plt.xticks(np.arange(len(show_cols)), show_cols, rotation=35, ha='right')
    plt.title("各方法综合指标归一化热力图")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_episode_trend(episode_data, metric, title, ylabel, out_path, focus_methods=None):
    plt.figure(figsize=(9.5, 5.6))
    methods = list(episode_data.keys())
    if focus_methods is not None:
        methods = [m for m in methods if m in focus_methods]
    for m in methods:
        x = episode_data[m]["episode"]
        y = episode_data[m][metric]
        plt.plot(x, y, marker='o', linewidth=1.5, markersize=3, label=m)
    plt.xlabel("训练轮次（Episode）")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_key_method_compare(episode_data, out_path):
    key_methods = [m for m in ["DQN", "PPO", "RA-DDQN", "SemiGreedy"] if m in episode_data]
    plt.figure(figsize=(10.0, 6.0))
    for m in key_methods:
        x = episode_data[m]["episode"]
        y = episode_data[m]["success_rate"]
        plt.plot(x, y, marker='o', linewidth=1.8, markersize=3.5, label=f"{m} 成功率")
    plt.xlabel("训练轮次（Episode）")
    plt.ylabel("成功率")
    plt.title("关键方法成功率训练趋势对比")
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def write_readme(out_dir):
    text = """
输出文件说明：

1. final_总奖励.png
   各方法最终总奖励对比。适合展示综合优化目标。

2. final_成功率.png
   各方法最终成功率对比。最直观反映任务完成效果。

3. final_平均响应时间.png
   各方法平均响应时间，越低越好。

4. final_平均成本.png
   各方法平均成本，越低越好。

5. final_恶意节点分配次数.png
   各方法将请求分配给恶意 Oracle 的次数，越低越安全。

6. final_可信节点分配次数.png
   各方法将请求分配给可信 Oracle 的次数，越高越好。

7. final_节点类型分配构成.png
   恶意 / 普通 / 可信节点分配的堆叠柱状图，便于整体比较。

8. final_综合指标热力图.png
   将各项指标归一化后形成热力图，可整体展示各方法优劣。

9. trend_成功率训练趋势.png
   各方法逐 Episode 成功率变化趋势。

10. trend_总奖励训练趋势.png
    各方法逐 Episode 总奖励变化趋势。

11. trend_DQN_PPO_RA对比.png
    DQN / PPO / RA-DDQN / SemiGreedy 的关键趋势对比图。

12. summary_results.csv
    Final results 原始汇总表。

13. summary_normalized.csv
    归一化后的汇总表，含综合得分。
"""
    with open(Path(out_dir) / "README_图表说明.txt", "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="运行日志 txt 路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    set_chinese_font()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    stem = input_path.stem
    if args.output_dir is None:
        out_dir = input_path.parent / f"{stem}_paper_figs"
    else:
        out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    text = input_path.read_text(encoding="utf-8", errors="ignore")

    df = build_final_dataframe(text)
    episode_data = parse_episode_metrics(text)
    norm_df = save_dataframe(df, out_dir)

    # 最终结果类图
    _bar_plot(df, "方法", "总奖励", "各方法最终总奖励对比", "总奖励", Path(out_dir) / "final_总奖励.png", "{:.3f}")
    _bar_plot(df, "方法", "成功率", "各方法最终成功率对比", "成功率", Path(out_dir) / "final_成功率.png", "{:.3f}")
    _bar_plot(df, "方法", "平均响应时间", "各方法最终平均响应时间对比", "平均响应时间", Path(out_dir) / "final_平均响应时间.png", "{:.3f}")
    _bar_plot(df, "方法", "平均成本", "各方法最终平均成本对比", "平均成本", Path(out_dir) / "final_平均成本.png", "{:.3f}")
    _bar_plot(df, "方法", "分配到恶意节点次数", "各方法分配到恶意节点次数对比", "分配次数", Path(out_dir) / "final_恶意节点分配次数.png", "{:.0f}")
    _bar_plot(df, "方法", "分配到可信节点次数", "各方法分配到可信节点次数对比", "分配次数", Path(out_dir) / "final_可信节点分配次数.png", "{:.0f}")

    plot_stacked_assignment(df, Path(out_dir) / "final_节点类型分配构成.png")
    plot_heatmap(norm_df, Path(out_dir) / "final_综合指标热力图.png")

    # 训练趋势类图
    if episode_data:
        plot_episode_trend(episode_data, "success_rate", "各方法成功率训练趋势", "成功率", Path(out_dir) / "trend_成功率训练趋势.png")
        plot_episode_trend(episode_data, "reward", "各方法总奖励训练趋势", "总奖励", Path(out_dir) / "trend_总奖励训练趋势.png")
        plot_key_method_compare(episode_data, Path(out_dir) / "trend_DQN_PPO_RA对比.png")

    write_readme(out_dir)

    print("绘图完成。输出目录：", out_dir)
    print("已生成文件：")
    for p in sorted(out_dir.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
