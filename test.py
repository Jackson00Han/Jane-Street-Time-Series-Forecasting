from pipeline.io import fs, P, cfg, storage_options
paths = fs.glob(f"az://jackson/js_exp/raw/train.parquet/partition_id=[4-6]/*.parquet")
print(len(paths), paths[:3])
