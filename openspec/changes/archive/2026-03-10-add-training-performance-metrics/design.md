## Context

The current trainer already has a clean logging boundary: every `log_freq` optimizer steps it converts accumulated loss statistics into scalar metrics and emits them through `accelerator.log(...)`. That path exposes optimization state, but it does not describe how fast training is progressing or whether the trainer is waiting on data delivery versus model execution.

YAVLA also already has an offline dataloader benchmark in `scripts/bench_dataloader.py` for end-to-end loader throughput and optional transform timing. That tool is useful for isolated benchmarking, but it does not answer the training-time question of whether a real run is compute-bound or input-bound.

This change should stay small and local to the trainer. It should add performance visibility without introducing new dependencies, worker-side instrumentation, or a new config surface.

## Goals / Non-Goals

**Goals:**
- Add concise training-time performance metrics to the existing trainer logging path.
- Define the metrics at the optimizer-step boundary so they remain meaningful with gradient accumulation.
- Distinguish trainer-visible data wait from the remaining host-side training time.
- Reuse the existing `log_freq` cadence and W&B integration.
- Keep the implementation small enough to fit naturally into `Trainer.run()` and the surrounding tests.

**Non-Goals:**
- Per-transform latency instrumentation inside training.
- GPU memory, utilization, or profiler trace logging.
- New CLI flags or config fields for performance logging.
- Replacing the standalone dataloader benchmark.
- Fine-grained worker-process timing or PyTorch profiler integration in the training loop.

## Decisions

### D1: Measure performance at the optimizer-step boundary

**Choice:** The trainer will measure performance per optimizer step, not per micro-batch. One optimizer step includes all micro-batches consumed under `gradient_accumulation_steps`.

**Why:** YAVLA's training loop already treats optimizer steps as the authoritative unit for logging, checkpointing, and progress. Measuring at that same boundary keeps throughput aligned with `train/global_batch_size` and avoids producing misleading metrics when accumulation is enabled.

**Alternative considered:** Measure every micro-batch. Rejected because the resulting metrics would not line up with the trainer's public notion of progress and would make throughput interpretation harder.

### D2: Define dataloader latency as trainer-visible wait time around batch fetch

**Choice:** `data_wait_time` will be measured as host wall-clock time spent blocked while fetching the next micro-batch from the training iterator.

**Why:** This is the simplest reliable definition that answers the operational question, "Is training stalled on input delivery?" It naturally includes dataloader, queue, decode, and collate delays that are visible to the trainer, while allowing prefetching to hide work when the pipeline keeps up.

**Alternative considered:** Instrument dataset transforms or worker internals. Rejected because it is more invasive, breaks down with multiple workers, and duplicates the scope of `scripts/bench_dataloader.py`.

### D3: Emit a small fixed metric set under a dedicated `perf/` namespace

**Choice:** The trainer will log:
- `perf/step_time_s`
- `perf/samples_per_sec`
- `perf/data_wait_time_s`
- `perf/compute_time_s`
- `perf/data_wait_fraction`

`compute_time_s` is defined as the non-wait remainder of trainer step time. `data_wait_fraction` is the fraction of step wall time attributed to waiting for input.

**Why:** This is the minimum set that makes the run diagnosable without turning the trainer into a profiler. The names are explicit and separate from optimization metrics already logged under `train/`.

**Alternative considered:** Add GPU memory, utilization, or per-stage sub-timings. Rejected to keep scope small and implementation concise.

### D4: Aggregate performance metrics over each logging window

**Choice:** Performance metrics will be accumulated across optimizer steps since the previous log event, then emitted once at the existing `log_freq` boundary.

**Why:** Single-step timing is noisy. Windowed logging produces more stable throughput and wait-fraction signals while fitting the existing logging contract. This also keeps the runtime overhead negligible.

**Alternative considered:** Emit only the most recent step's timing values. Rejected because it would make `log_freq > 1` performance logs too noisy and less representative.

### D5: Preserve the current config surface and logging behavior

**Choice:** Performance metrics will piggyback on the existing logging flow. No new config flag will be introduced. When W&B logging is enabled, the new `perf/*` metrics are sent alongside the existing `train/*` metrics. Console logging may remain concise and does not need to print every new metric.

**Why:** This avoids unnecessary configuration sprawl and matches the user's request for a concise implementation.

**Alternative considered:** Add a dedicated `performance_logging` config block or always print performance metrics to console. Rejected because the current logging controls are already sufficient.

## Risks / Trade-offs

- **[Trainer-visible wait is not full pipeline attribution]** → Document that `perf/data_wait_time_s` measures stall observed by the trainer, not per-worker transform cost. Keep `scripts/bench_dataloader.py` as the deeper profiling tool.
- **[Distributed runs can hide per-rank skew]** → Start with the trainer's existing logging model and keep semantics clear in docs. If rank-to-rank timing skew becomes important, add distributed reduction in a follow-up change.
- **[Windowed metrics hide short spikes]** → Use the existing `log_freq` knob; users who need finer timing visibility can lower it.
- **[Additional timing code could drift from the training loop]** → Keep the instrumentation close to batch fetch and optimizer-step boundaries, with targeted tests for gradient accumulation and cadence behavior.
