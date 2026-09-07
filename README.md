# NeuralAtmosphereOperator

Global 0.5-degree ERA5 forecasting with NVIDIA's Spherical Fourier Neural
Operator (SFNO). The selected corpus uses twenty-four training years and 26
prognostic channels; the compact model defaults to SFNO-SC2-L6-E128 with all
180 modes of its internal 181x360 spherical grid retained.

For the audited environment, install the exact versions in
`requirements-lock.txt`. In particular, the model is locked to
`torch-harmonics==0.7.4`; changing that package requires repeating the model,
parameter-count, checkpoint and rollout tests.

## Expected data layout

```text
data/dataset/
├── era5_1995_2020_0p5deg_26ch.zarr
└── stats/
    ├── means.npy
    ├── stds.npy
    ├── time_means.npy
    ├── time_diff_means_dt1.npy
    ├── time_diff_stds_dt1.npy
    ├── stats.json
    └── channels.txt
```

Download/regrid in restartable seven-day batches, then compute statistics from
the logical training range only:

```bash
python data/test_download_sample.py --output data/dataset/era5_sample_audited.zarr
python data/download_data.py --confirm-full-download
python scripts/compute_stats.py --time-step 1
python scripts/validate_data.py --full-scan --output runs/data_validation.json
python scripts/check_backend.py --device auto
python scripts/audit_training_readiness.py --sample-zarr data/dataset/era5_sample_audited.zarr
python scripts/benchmark_sample.py --device auto --height 121 --width 240
python scripts/estimate_vram.py --batch-size 4 --gpu-vram-gib 24
python scripts/benchmark_model.py --batch-size 4 --rollout-steps 1
```

The downloader fingerprints the source, dates, cadence, resolution and ordered
channel selection. A legacy `.partial` store or sample created before this
contract was introduced must be preserved under another name or regenerated;
it will not be resumed silently. Flattened stores also carry units and source
metadata separately for every channel.

One physical Zarr store is sliced chronologically: training is 1995--2018,
validation is the full year 2019, and the untouched test year is 2020. This
provides 35,064 / 1,460 / 1,464 states respectively. Default training selection
uses 32 evenly spaced validation initializations with 60 leads. The same panel
must be used for all candidate models; final evaluation should cover more
initializations and report uncertainty across weather episodes.

Statistics are streamed in float64 with bounded memory, written as immutable
version-4 bundles, and checked by artifact hashes, coordinates, units, cadence
and training interval. Existing statistics must be recomputed into a **new**
directory. The old 1995--early-2019 store is insufficient for this protocol;
do not rename it to satisfy the new filename. Preserve it and download the
missing periods under a new download contract.

`audit_training_readiness.py` projects both the compressed download and peak
disk usage from the local one-day sample. `estimate_vram.py` is intentionally a
conservative analytical planner; `preflight_a100.py` measures actual optimizer updates and long validation on
the target GPU; a analytical planner or random-loss benchmark is insufficient.

## Recommended training curriculum

The default budget is 25 epochs. `--epochs 0` explicitly opts into an unlimited
run; it is not the default. Use a fixed budget for comparisons, and choose
checkpoints using validation only. See [the A100 runbook](docs/A100_TRAINING.md)
for data verification, actual CUDA preflight, and launch commands.

Start with one-step training. Increase rollout length only after convergence,
initializing a new optimizer stage from the previous best weights:

```bash
# Stage 1: stable one-step objective
python scripts/train.py \
  --run-dir runs/sfno_stage1 \
  --stage-name sfno_1step \
  --rollout-steps 1 \
  --validation-rollout-steps 60 \
  --validation-batch-size 1 \
  --batch-size 4 \
  --gradient-accumulation 2 \
  --num-workers 4 --prefetch-factor 2 --no-persistent-workers \
  --no-gradient-checkpointing

# Stage 2: short autoregressive fine-tuning
python scripts/train.py \
  --run-dir runs/sfno_stage2 \
  --stage-name sfno_2step \
  --init-checkpoint runs/sfno_stage1/checkpoints/best.pt \
  --rollout-steps 2 \
  --validation-rollout-steps 60 \
  --validation-batch-size 1 \
  --gradient-checkpointing \
  --batch-size 4 \
  --gradient-accumulation 2 \
  --epochs 5 --warmup-epochs 1 --patience 3 \
  --learning-rate 1e-4
```

Use `--resume` only to continue the exact same numerical trajectory. Use
`--init-checkpoint` for a new curriculum stage; optimizer, scheduler and early
stopping state intentionally restart. A version-2 epoch-end checkpoint stores
the model, complete AdamW state (`step`, `exp_avg`, `exp_avg_sq` and parameter
groups), learning-rate scheduler, AMP loss scaler, completed-update counter,
early-stopping state, Python/NumPy/PyTorch RNG and the training-loader generator.
It is written atomically through `*.tmp` and then renamed. An interruption in
the middle of an epoch therefore returns to the end of the last completed
epoch, never to a partially accumulated gradient or a half-written checkpoint.

The training loader asynchronously prepares up to `num_workers *
prefetch_factor` batches while the accelerator processes the current batch.
CUDA runs use pinned host memory and non-blocking host-to-device copies. The
epoch log reports `data_wait`; sustained values above roughly 15--20% indicate
that storage/decompression is starving the accelerator. Training workers are
recreated at epoch boundaries because persistent worker state is not captured
by a checkpoint. Multiprocess prefetching remains active; this small process-
startup cost is required for an exact epoch-boundary resume.

The same epoch log records the mean/max pre-clipping gradient norm, clipping
fraction and non-finite count. Treat sustained clipping near 100% as evidence
that `--gradient-clip 1.0` is too restrictive; decide from the real-data pilot
rather than changing this threshold speculatively.

The primary objective follows the official Makani SFNO recipe more closely than
plain normalized MSE:

- exact spherical-cell-area weighting on the equiangular grid;
- Makani `auto` channel priorities (including pressure-level weights);
- temporal-difference scaling `global_std / time_diff_std`, computed from the
  training split at the configured forecast stride;
- loss at every autoregressive lead and final-lead-aware checkpoint selection.

Evaluation computes a spatial ACC for each forecast initialization and channel
after subtracting the training-only long-term spatial climatology in
`time_means.npy`, then averages over initializations. This is the SFNO-style
long-term-climatology definition, not WeatherBench2's day-of-year/time-of-day
climatology.

Spectral loss, channel-relative loss, input noise, and the NeuralOceanOperator
per-channel backward `LossScaler` are available as explicit ablations. They are
disabled by default because combining them with Makani's channel weighting
changes the official baseline objective. Enable only after a controlled
lead-wise comparison, for example `--use-loss-scaler` or
`--input-noise-std 0.01`.

The optimizer defaults are conservative for a single accelerator (AdamW,
update-level warmup/cosine decay, accumulation, clipping and BF16). The locked
compact model uses SC2 to retain 180 spherical modes on the 0.5-degree grid,
with Makani's instance normalization and Driscoll–Healy operator, 6 layers and
width 128. Its analytical capacity is 35,793,920 real-scalar degrees of freedom
(18,099,200 PyTorch tensor elements because the spectral weights are complex).
An E384/L8 capacity ablation remains available through the corresponding
training CLI overrides. It retains this project's SC2/26-channel contract and
must not be described as an exact Makani reproduction. The WB2-native
26-channel contract preserves
the key 250-hPa jet level, 50-hPa geopotential, lower-tropospheric moisture and
total-column water vapour. It is an explicitly documented approximation of
the paper's compact set because this WB2 archive does not contain 100-m winds.

## Evaluation and inference

```bash
python scripts/evaluate.py \
  --checkpoint runs/sfno_stage2/checkpoints/best.pt \
  --split valid --rollout-steps 60 --report-every-days 3

# Final untouched 31-day period; run only after model selection is frozen.
python scripts/evaluate.py \
  --checkpoint runs/sfno_stage2/checkpoints/best.pt \
  --split test --rollout-steps 120 --report-every-days 3

python scripts/rollout.py \
  --checkpoint runs/sfno_stage2/checkpoints/best.pt \
  --rollout-steps 120 --samples 1

python scripts/plot_results.py --run-dir runs/sfno_stage2
```

Checkpoints store the model/training configuration, optimizer, update-level
scheduler, AMP scaler, process RNGs, DataLoader generator, dataset signature,
normalization fingerprints (including temporal differences), and exact
PyTorch/torch-harmonics runtime fingerprint. Timestamps are required to be
strictly increasing with a regular cadence; reported lead hours are derived
from the data rather than hard-coded.

Primary references: [SFNO paper](https://proceedings.mlr.press/v202/bonev23a.html),
[NVIDIA Makani SFNO configuration](https://github.com/NVIDIA/makani/blob/main/config/sfnonet.yaml),
[Makani data statistics guide](https://github.com/NVIDIA/makani/blob/main/data_process/Readme.md),
and [torch-harmonics](https://github.com/NVIDIA/torch-harmonics).

## Audit corrections before accelerator training

The original [scientific audit](audit/2026-09-06/SCIENTIFIC_AUDIT_REPORT.md) describes
the pre-fix snapshot. The [remediation ledger](docs/AUDIT_REMEDIATION.md) records
implemented fixes, tests and outstanding experiments. Complex degree kernels
share nonnegative harmonic orders; this is not a guarantee of full SO(3)
equivariance. Scalar wind channels and absent external forcing remain modeling
approximations to investigate through controlled experiments.

Evaluation rejects grid/cadence/unit mismatches and overlap with checkpoint
training or validation periods even when data paths are overridden. It reports
persistence and training-climatology baselines, per-channel physical bias,
spatial variance ratio and ACC valid counts. Output directories with completed
results are not overwritten. `--allow-data-mismatch` only marks diagnostic
statistics changes; it does not permit wrong geometry or held-out leakage.

For deployment after the last observed timestamp, use `rollout.py --forecast-only`
with `--start-index` selecting the desired initialization; no future truth is
required. Forecast Zarr outputs contain explicit initial and valid times.

The forward SHT additionally separates the constant component analytically to
prevent float32 quadrature leakage amplified by InstanceNorm. This preserves
constants on the tested full CPU grid without adding parameters; numerical
results can differ from unmodified torch-harmonics. Near-constant sensitivity
and CUDA behavior require the preflight/pilot checks in the A100 runbook.
