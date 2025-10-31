# Don’t Force LLMs—First Nail the Data → Features → Training Pipeline

Below is a 6-layer, plug-and-play stack for large-scale training that’s **practical, general-purpose, and stable**. At the end is a **TFT recipe** for wide multivariate time series.

## 1) Data & Pipelines (I/O sets the ceiling)
- **Time/hash sharding:** Parquet/Arrow; shard by day or week + column pruning; **100–500 MB** per shard is ideal.  
- **Column pruning & predicate pushdown:** use Polars/Arrow `.select()` / `.filter()` to read only needed cols/rows.  
- **Precompute heavy work:** z-score, aggregations, lags, etc., offline; write back per-shard (your day→week merge is good).  
- **Async prefetch:** `DataLoader(num_workers, prefetch_factor)`; parse/decompress on CPU → train on GPU.  
- **Cache & retry:** keep hot weekly shards on local NVMe; auto-retry on failures to smooth throughput.

## 2) Feature Management (first filter, then compress)
- **Auto filtering:** drop constants/low variance; for high correlation (**|ρ| > 0.98**), keep one; rank by LightGBM/XGBoost importance and take **Top-K**.  
- **Unsupervised compression:** group “long-tail weak” features → PCA/AutoEncoder down to **8–32 dims**; keep strong signals raw.  
- **Hashed embeddings:** for ultra-high-cardinality categoricals, use hashing + embedding to avoid giant vocab tables.  
- **Online selection:** with TFT’s VSN, do a short all-feature run → read variable importance → slim down → retrain.

## 3) Memory & Speed (the value trio)
- **Mixed precision:** **bf16** preferred (more stable) or fp16; enable GradScaler.  
- **Grad accumulation & micro-batches:** `accumulate_grad_batches` × small `batch_size` → effective large batch.  
- **Activation checkpointing:** enable gradient/activation recompute to cut memory with minimal overhead.  
- **Bonus:** FlashAttention/xFormers (attention), `torch.compile` (PyTorch 2), CUDA Graphs (after things are stable).

## 4) Parallelism & Optimizers (scale pays off)
- **FSDP or DeepSpeed ZeRO-3:** shard params/grads/optimizer—your go-to for big models.  
- **Offload when tight:** CPU/NVMe offload (ZeRO-Offload or FSDP+CPU). Throughput drops, but training fits.  
- **Hybrid parallel:** DP + (if truly huge) pipeline/tensor parallel.  
- **Optimizers:** **AdamW** is the safe default; for ultra-large models/long sequences try **Adafactor** (memory-lean) or **Lion** (fast).

## 5) Stability & Generalization (big data still needs discipline)
- **LR schedule:** warmup + cosine / one-cycle; **grad clip = 1.0**.  
- **Regularization:** dropout, early stopping, weight decay, **SWA**.  
- **Monitoring:** peak memory, throughput, dead-activation rate, gradient norms, VSN importances; watch for all-zero/constant columns.

## 6) Sequence Modeling (time-series specifics)
- **Shrink “effective length × batch”:** memory scales with **enc_len × batch_size**—tune both first.  
- **Efficient attention:** for long sequences, use FlashAttention or sparse/kernel methods (Performer/Longformer families).  
- **Grouped modeling:** with many symbols/items, train light models per cluster/sector, then fuse predictions.

---

# Practical Recipe: Wide Multivariate Time Series with **TFT**
1. **Storage & ingest**  
   - Write time-/hash-sharded **Parquet/Arrow** (daily/weekly, 100–500 MB).  
   - Precompute: z-scores per group, rolling means/std, lags/leads, calendar flags, target encodings (leak-safe).  
   - Polars pipeline: `.scan_parquet()` → `.filter(time range & mask)` → `.select(required cols)`.

2. **Feature pass-1 (filter/shape)**  
   - Drop constants/near-zero-variance; collapse |ρ|>0.98.  
   - Train a quick **LightGBM/XGBoost** on a sample → keep **Top-K** numeric/categorical drivers.  
   - Long-tail numerics → **PCA (8–32)** per family; very high-cardinality categoricals → **hash + embedding**.

3. **TFT setup**  
   - Keep strong raw features + compressed components; map categoricals to hashed IDs.  
   - Tune **encoder length** and **batch size** to fit memory; use **bf16**, GradScaler, activation checkpointing.  
   - Enable **FlashAttention/xFormers** if available; use `torch.compile` once stable.

4. **Training loop**  
   - **Optimizer:** AdamW; **LR:** warmup → cosine; **grad clip 1.0**; dropout/weight decay modest.  
   - **Accumulation:** use micro-batches + gradient accumulation to hit effective target batch.  
   - **Early stop** on validation sMAPE/RMSE; keep **SWA** checkpoint.

5. **Feature pass-2 (VSN-guided slimming)**  
   - Short run with all candidates → read **VSN variable importances**.  
   - Prune weak variables; optionally expand embeddings for top categoricals; **retrain final**.

6. **Serving & ops**  
   - Export preprocessing graph (Polars → Arrow) + model; cache **hot weekly shards** on NVMe.  
   - Add **retry logic** and **telemetry** (throughput, memory, grad norms, importances).  
   - For many symbols: train per-cluster TFTs and **blend** (e.g., weighted average or meta-learner).
