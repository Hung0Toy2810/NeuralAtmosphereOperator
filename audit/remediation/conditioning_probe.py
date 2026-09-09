import sys, json, time, resource, argparse
from pathlib import Path

sys.path[:0] = [".", "src"]
import torch
from configs.model_config import AtmosphereModelConfig
from neural_atmosphere_operator.models.model import AtmosphereNeuralOperator
from dataclasses import asdict

parser = argparse.ArgumentParser()
parser.add_argument("--native", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
torch.set_num_threads(1)
torch.manual_seed(42)
config = AtmosphereModelConfig(stabilize_sht_constants=not args.native)
model = AtmosphereNeuralOperator(config).eval()
result = {
    "device": "cpu",
    "torch": torch.__version__,
    "config": asdict(config),
    "seed": 42,
    "probes": {},
}
with torch.inference_mode():
    for name in ("zero", "constant", "near_constant"):
        start = time.perf_counter()
        x = (
            torch.zeros(1, config.in_channels, *config.img_size)
            if name == "zero"
            else torch.ones(1, config.in_channels, *config.img_size)
        )
        if name == "near_constant":
            x = x + 1e-5 * torch.randn_like(x)
        y = model(x)
        result["probes"][name] = {
            "finite": bool(torch.isfinite(y).all()),
            "spatial_std_mean": float(y.std(dim=(-2, -1)).mean()),
            "rms": float(y.square().mean().sqrt()),
            "seconds": time.perf_counter() - start,
        }
        print(name, result["probes"][name], flush=True)
result["peak_process_rss_gib"] = (
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    * (1 if sys.platform == "darwin" else 1024)
    / 2**30
)
args.output.write_text(json.dumps(result, indent=2) + "\n")
