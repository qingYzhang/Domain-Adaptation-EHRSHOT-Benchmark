import os
import json
import pandas as pd

# 四个版本目录
versions = ["EHRSHOT-Benchmark-nhird-format_qwen_70",
            "EHRSHOT-Benchmark-nhird-format_qwen_80",
            "EHRSHOT-Benchmark-nhird-format_qwen_baseline",
            "EHRSHOT-Benchmark-nhird-format_qwen",
            "./"
            ]

# benchmark 下的任务子文件夹
tasks = [
    # 'guo_los',
    # 'new_hypertension',
    # 'new_acutemi',
    # 'new_hyperlipidemia',
    # 'guo_readmission',
    # 'new_celiac',
    'new_pancan'
]

# 输出目录
output_dir = "merged_results"
os.makedirs(output_dir, exist_ok=True)

for metric in ["AUROC", "AUPRC"]:
    for task in tasks:
        rows = {}
        for v in versions:
            version_name = v.split("_")[-1]  # v1, v2...
            json_path = os.path.join(v, "benchmark", task, "all_shot_scores_160m_MaxAbsScaler.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, "r") as f:
                data = json.load(f)
            
            row = {}
            for k, scores in data.items():
                if k == "-1":
                    row["All"] = scores.get(metric, None)
                else:
                    row[str(k)] = scores.get(metric, None)
            rows[version_name] = row

        if not rows:
            continue

        # 转为 DataFrame
        df = pd.DataFrame(rows).T  # 行=版本, 列=shot
        # 调整列顺序
        numeric_cols = sorted([int(c) for c in df.columns if c not in ["All"]])
        cols = ["All"] + [str(k) for k in numeric_cols]
        df = df[cols]
        df.index.name = task  
        # 保存 CSV
        out_path = os.path.join(output_dir, f"{task}_{metric}.csv")
        df.to_csv(out_path)

"Finished merging results with All = -1."
