# run_feature_select.py
import polars as pl

from pipeline.io import cfg, P, ensure_dir_local
from pipeline.memmap import make_sliding_cv_fast
from pipeline.io import cfg, fs, storage_options, P, ensure_dir_local, ensure_dir_az

def main():
    test_path = "az://jackson/js_exp/exp/v1/panel_shards/panel_1080_1109.parquet"
    lf = pl.scan_parquet(test_path, storage_options=storage_options)
    
    lf.limit(5).collect()
if __name__ == "__main__":
    main()
