    # =========================
    # Based on CV results, Refitting on full window & saving model
    # =========================
    final_dir = os.path.join(model_dir, "final"); ensure_dir_local(final_dir)
    final_rounds = max(50, int(np.mean(best_iters) * 1.10))  # 1.10 为安全放大系数
    print(f"[fixed] refit on full window with num_boost_round={final_rounds}")

    d_full = d_base  # 全部训练窗
    bst_final = lgb.train(
        {**fixed_params, "verbosity": -1},
        d_full,
        feval=lgb_wr2,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(period=log_period)],
    )

    model_path = os.path.join(final_dir, f"lgbm_fixed__{tag}.txt")
    bst_final.save_model(model_path)
    print(f"[fixed][done] model -> {model_path}")

    meta_path = os.path.join(final_dir, f"lgbm_fixed__{tag}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "memmap_prefix": path_mm,
            "train_range": [int(lo), int(hi)],
            "params": fixed_params,
            "num_boost_round": final_rounds,
            "seed": seed_val,
            "features": feat_cols,
            "reports": {
                "fi_csv": fi_path,
                "summary_json": out_path,
                "validation_csv": rank_path,
            },
            "tag": tag,
        }, f, indent=2)
    print(f"[fixed][done] model meta -> {meta_path}")