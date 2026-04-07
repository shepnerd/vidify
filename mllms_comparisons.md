# MLLM Comparison: Qwen3.5 vs Qwen3-MLA

**Date:** 2026-04-06

## Models

| | Qwen3.5 (A) | Qwen3-MLA (B) |
|---|---|---|
| **Model** | Qwen3.5-9B | hf-5615 (Qwen3-VL + MLA attention) |
| **Checkpoint** | `Qwen/Qwen3.5-9B` | `/mnt/shared-storage-gpfs2/sfteval/xtuner_saved_model/internvl3.5/ablate_wuyue2/20260331093205/hf-5615` |
| **Backend** | vLLM (optimized) | HuggingFace Transformers (naive) |
| **GPUs** | 4 | 1 |
| **Port** | 8000 | 8001 |
| **Attention** | GQA (Grouped Query Attention) | MLA (Multi-head Latent Attention) |

## Benchmark Results

### Text-only Prompt

Prompt: "请列举中国四大名著及其作者，每本书用一句话概括主要内容。"

| Metric | Qwen3.5 (A) | Qwen3-MLA (B) | Ratio |
|--------|-------------|----------------|-------|
| Avg Latency | 0.757s | 2.614s | A 3.45x faster |
| Avg Throughput | 338.2 tok/s | 44.0 tok/s | A 7.69x faster |
| Avg Tokens | 256.0 | 115.0 | — |
| Runs | 3 | 3 | — |

### Image Prompt

Prompt: "请描述这张图片中的内容。" (frame extracted from `taste_in_china_s1e1.mp4` at t=10s)

| Metric | Qwen3.5 (A) | Qwen3-MLA (B) | Ratio |
|--------|-------------|----------------|-------|
| Avg Latency | 0.839s | 1.592s | A 1.90x faster |
| Avg Throughput | 305.3 tok/s | 26.2 tok/s | A 11.65x faster |
| Avg Tokens | 256.0 | 41.7 | — |
| Runs | 3 | 3 | — |

### Video Prompt

Skipped — MLA server on 1 GPU OOM'd when processing video frames alongside the model.

## Caveats

1. **Not an apples-to-apples comparison.** Qwen3.5 uses vLLM (continuous batching, PagedAttention, KV cache optimization) on 4 GPUs. Qwen3-MLA uses raw HuggingFace Transformers (naive autoregressive generation) on 1 GPU.
2. **MLA's real advantage is KV-cache compression** via low-rank factorization (`kv_a_proj_with_mqa` + `kv_b_proj`), which reduces memory per token. This matters most at long context / high throughput — something vLLM would exploit but the transformers backend does not.
3. **Token count difference.** Qwen3.5 consistently generates 256 tokens (hitting `max_tokens`), while MLA generates fewer (41–128). This may reflect different generation behavior or model calibration, and affects throughput comparisons.
4. **vLLM cannot serve MLA** due to incompatible attention weight names (`kv_a_proj_with_mqa`/`kv_b_proj` vs expected `q_proj`/`k_proj`/`v_proj` + `q_norm`/`k_norm`). A fair comparison would require either a vLLM plugin for MLA or custom CUDA kernels.

## How to Reproduce

```bash
# Terminal 1: Qwen3.5 (vLLM, 4 GPUs)
set -a && source .env && set +a
bash scripts/run_test_gpu.sh --model qwen3.5

# Terminal 2: Qwen3-MLA (transformers, 1 GPU)
set -a && source .env && set +a
bash scripts/rl.sh -gpu 1 -- bash scripts/serving_qwen3_mla.sh

# Terminal 3: Run comparison
python scripts/compare_models.py \
    --api-base-a http://<qwen35-ip>:8000/v1 \
    --api-base-b http://<mla-ip>:8001/v1 \
    --video media/taste_in_china_s1e1.mp4 \
    --runs 3 \
    --skip-video  # add if MLA on single GPU
```
