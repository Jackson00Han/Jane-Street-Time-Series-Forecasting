# Cloud
jackson/js_exp/
  raw/              # 原始分片（只追加，不改写）
  clean/            # 规整后的“金主表”（只追加/换新版本，不改写）
  exp/v1/
    fe_shards/      # 特征分片（发布后视为只读）
    models/         # 训练产物（按 run 号归档，写一次不覆盖）
    reports/        # 指标/图表/日志归档
    registry.json   # 数据/模型清单（行数、schema、时间范围…）

# local

/mnt/data/js/
  exp/v1/
    mm/             # memmap，仅本地
    cache/          # 可重建缓存（删了能再算）
    tmp/            # 临时文件（自动清）
