# NeuralAtmosphereOperator

Global 0.5-degree ERA5 forecasting with NVIDIA's Spherical Fourier Neural
Operator (SFNO). The selected corpus uses twenty-four training years and 71
prognostic channels; the compact model defaults to SFNO-SC2-L6-E192 with all
180 modes of its internal 181x360 spherical grid retained.

For the audited environment, install the exact versions in
`requirements-lock.txt`. In particular, the model is locked to
`torch-harmonics==0.7.4`; changing that package requires repeating the model,
parameter-count, checkpoint and rollout tests.

## Expected data layout

```text
data/dataset/
├── era5_1995_2020_0p5deg_71ch.zarr
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
python data/test_download_sample.py --output data/dataset/era5_sample_0p5_71ch.zarr
python data/download_data.py --confirm-full-download
python scripts/compute_stats.py --time-step 1
python scripts/validate_data.py --full-scan --output runs/data_validation.json
python scripts/check_backend.py --device auto
python scripts/audit_training_readiness.py --sample-zarr data/dataset/era5_sample_0p5_71ch.zarr
python scripts/benchmark_sample.py --device auto --height 121 --width 240
python scripts/estimate_vram.py --batch-size 1 --gpu-vram-gib 32
python scripts/benchmark_model.py --batch-size 1 --rollout-steps 1
```

The downloader fingerprints the source, dates, cadence, resolution and fixed
71-channel order. It exposes no variable or pressure-level selection. A legacy
`.partial` store or sample created before this
contract was introduced must be preserved under another name or regenerated;
it will not be resumed silently. Flattened stores also carry units and source
metadata separately for every channel.

One physical Zarr store is sliced chronologically: training is 1995--2018,
validation is the full year 2019, and the untouched test year is 2020. This
provides 35,064 / 1,460 / 1,464 states respectively. Default training selection
uses 32 evenly spaced validation initializations with eight free-running leads
(6--48 hours). The same panel must be used for all candidate models; final
evaluation should cover more initializations and report uncertainty across
weather episodes.

Statistics are streamed in float64 with bounded memory, written as immutable
version-4 bundles, and checked by artifact hashes, coordinates, units, cadence
and training interval. Existing statistics must be recomputed into a **new**
directory. The old 1995--early-2019 store is insufficient for this protocol;
do not rename it to satisfy the new filename. Preserve it and download the
missing periods under a new download contract.

`audit_training_readiness.py` projects both the compressed download and peak
disk usage from the local one-day sample. `estimate_vram.py` is intentionally a
conservative analytical planner; `preflight_gpu.py` measures actual optimizer
updates and stage-matched validation on the selected CUDA GPU.

## Recommended training curriculum

The default budget is 25 epochs. `--epochs 0` explicitly opts into an unlimited
run; it is not the default. Use a fixed budget for comparisons, and choose
checkpoints using validation only. See [the GPU runbook](docs/GPU_TRAINING.md)
for data verification, actual CUDA preflight, and launch commands.

Training and validation always use the same horizon and the same loss. The
curriculum is K=1, 2, 4, 8; each later stage initializes from the preceding
stage's best checkpoint and starts a new optimizer with a lower learning rate.
At six-hour cadence these stages cover 6, 12, 24, and 48 hours respectively.
Validation scans every consecutive temporal window unless an explicit pilot
limit is supplied.

The finite 25-epoch budget is also protected by a default patience of five
validation rounds. With `--early-stopping-min-delta 0`, every strict improvement
resets patience and `best.pt` preserves the lowest stage-matched validation loss.

```bash
# Stage 1: stable one-step objective
python scripts/train.py \
  --run-dir runs/sfno_k1 \
  --stage-name sfno_1step \
  --rollout-steps 1 \
  --validation-batch-size 1 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --num-workers 4 --prefetch-factor 2 --no-persistent-workers \
  --no-gradient-checkpointing

# Stage 2: short autoregressive fine-tuning
python scripts/train.py \
  --run-dir runs/sfno_k2 \
  --stage-name sfno_2step \
  --init-checkpoint runs/sfno_k1/checkpoints/best.pt \
  --rollout-steps 2 \
  --validation-batch-size 1 \
  --gradient-checkpointing \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --epochs 10 --warmup-epochs 1 --patience 3 \
  --learning-rate 1e-4

# Stage 3: K=4 fine-tuning
python scripts/train.py \
  --run-dir runs/sfno_k4 --stage-name sfno_4step \
  --init-checkpoint runs/sfno_k2/checkpoints/best.pt \
  --rollout-steps 4 --gradient-checkpointing \
  --epochs 7 --warmup-epochs 0 --patience 3 \
  --learning-rate 5e-5

# Stage 4: K=8 final fine-tuning
python scripts/train.py \
  --run-dir runs/sfno_k8 --stage-name sfno_8step \
  --init-checkpoint runs/sfno_k4/checkpoints/best.pt \
  --rollout-steps 8 --gradient-checkpointing \
  --epochs 5 --warmup-epochs 0 --patience 3 \
  --learning-rate 2.5e-5
```

Use `--resume` only to continue the exact same numerical trajectory. Use
`--init-checkpoint` for a new curriculum stage; optimizer, scheduler and early
stopping state intentionally restart. A version-3 epoch-end checkpoint stores
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

The primary objective follows GraphCast's deterministic time-difference
standardization and SFNO's full autoregressive backpropagation:

- exact spherical-cell-area weighting on the equiangular grid;
- inverse time-difference variance scaling from the training split;
- variable-balanced pressure weights and documented surface weights;
- a uniform mean over exactly K leads, with gradients through the full rollout;
- the identical objective and K for train, validation, and test reporting.

Evaluation computes a spatial ACC for each forecast initialization and channel
after subtracting the training-only long-term spatial climatology in
`time_means.npy`, then averages over initializations. This is the SFNO-style
long-term-climatology definition, not WeatherBench2's day-of-year/time-of-day
climatology.

Input noise remains an explicit ablation. It is disabled for the baseline.

The optimizer defaults are conservative for a single accelerator (AdamW,
update-level warmup/cosine decay, accumulation, clipping and BF16). The locked
compact model uses SC2 to retain 180 spherical modes on the 0.5-degree grid,
with Makani's instance normalization and Driscoll–Healy operator, 6 layers and
width 192. Its capacity is 80,545,152 real-scalar degrees of freedom
(40,732,032 PyTorch tensor elements because the spectral weights are complex).
The repository provides no large-model preset: architecture overrides are
explicit CLI research controls rather than a supported alternative baseline.
The 71-channel contract includes all five core pressure-level variables
(u, v, geopotential, temperature and specific humidity) on all 13 WB2 levels,
plus six surface variables. The source lacks the two 100-m wind components
from the SFNO 73-channel reference. See [the source verification and channel
order](docs/DATA_CHANNELS.md). Recreate statistics and use a new run directory;
26-channel data/statistics and E128 checkpoints cannot initialize this model
through the current training CLI.

## Evaluation and inference

```bash
python scripts/evaluate.py \
  --checkpoint runs/sfno_k8/checkpoints/best.pt \
  --split valid --report-every-days 1

# Final untouched 2020 test year; run only after model selection is frozen.
python scripts/evaluate.py \
  --checkpoint runs/sfno_k8/checkpoints/best.pt \
  --split test --report-every-days 1

python scripts/rollout.py \
  --checkpoint runs/sfno_k8/checkpoints/best.pt \
  --rollout-steps 8 --samples 1

python scripts/plot_results.py --run-dir runs/sfno_k8
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
and CUDA behavior require the preflight/pilot checks in the GPU runbook.
