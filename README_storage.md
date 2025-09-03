/mnt/data/js                           # ← 本地“高速区”（最好挂独立数据盘，别用临时 /mnt）
├─ config/                              [GIT]
│   ├─ model_defaults.yaml
│   └─ fe_params.yaml
├─ exp/
│   └─ v1/
│       ├─ config/                      [GIT + 备份到 BLOB(可选)]
│       │   ├─ fe_params.yaml
│       │   ├─ model.yaml
│       │   ├─ data.yaml
│       │   └─ feature_sets/
│       │       ├─ r00_bootstrap.txt
│       │       ├─ r01_topk400.txt
│       │       ├─ r02_enriched_params.yaml
│       │       ├─ r03_corr_pruned.txt
│       │       └─ final_selected.txt
│       ├─ raw_shards/                  [BLOB]
│       ├─ fe_shards/                   [BLOB]
│       ├─ mm_shards/                   [BLOB]   # 训练前从这里读，构建本地 memmap
│       ├─ mm/                          [LOCAL]  # ★ 训练用 memmap（必须本地/数据盘）
│       ├─ models/                      [BLOB]   # 模型与 meta 持久化
│       ├─ reports/                     [BLOB]   # 指标/特征重要性
│       └─ tmp/                         [LOCAL]  # 临时目录（中间计算/缓存）
├─ raw/                                 [BLOB]   # 原始数据（只读备份）
└─ clean/                               [BLOB]   # 清洗后基表；需要时本地缓存一份
    └─ final_clean.parquet
cache/                                   [BLOB]  # 阶段性可复用的大缓存
└─ stage_c_{op}.parquet
