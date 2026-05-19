from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_torch.model import build_model, count_parameters, estimate_conv_macs, normalize_architecture, scaled_filters


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(x) for x in parse_csv(value)]


def parse_size(value: str) -> tuple[int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--input-size must be height,width")
    return parts[0], parts[1]


def benchmark_latency(model: torch.nn.Module, x: torch.Tensor, warmup: int, repeats: int) -> dict[str, float | int]:
    if repeats <= 0:
        return {"warmup": warmup, "repeats": repeats}

    device = x.device
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)

    return {
        "warmup": warmup,
        "repeats": repeats,
        "latency_ms_mean": statistics.mean(times) * 1000.0,
        "latency_ms_std": statistics.pstdev(times) * 1000.0,
        "fps_mean": 1.0 / statistics.mean(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute parameters, Conv2d MACs/FLOPs, and optional latency.")
    parser.add_argument("--architectures", default="light_unet,residual_unet,attention_unet,unetpp,inception_unet,se_unet,dense_unet")
    parser.add_argument("--width-multipliers", default="1.0")
    parser.add_argument("--input-size", default="256,256", help="height,width")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latency-warmup", default=10, type=int)
    parser.add_argument("--latency-repeats", default=0, type=int,
                        help="Set >0 to benchmark latency on the selected device.")
    parser.add_argument("--output", default=Path("outputs/model_complexity.csv"), type=Path)
    parser.add_argument("--json-output", default=None, type=Path)
    args = parser.parse_args()

    height, width = parse_size(args.input_size)
    device = torch.device(args.device)
    rows = []

    for architecture_arg in parse_csv(args.architectures):
        architecture = normalize_architecture(architecture_arg)
        for width_multiplier in parse_float_csv(args.width_multipliers):
            filters = scaled_filters(width_multiplier)
            model = build_model(architecture, in_channels=1, num_classes=6, filters=filters).to(device)
            params = count_parameters(model)
            macs = estimate_conv_macs(model, (args.batch_size, 1, height, width), device=device)
            x = torch.zeros((args.batch_size, 1, height, width), device=device)
            latency = benchmark_latency(model, x, args.latency_warmup, args.latency_repeats)
            row = {
                "architecture": architecture,
                "width_multiplier": width_multiplier,
                "filters": "x".join(str(x) for x in filters),
                "input_height": height,
                "input_width": width,
                "batch_size": args.batch_size,
                "parameter_count": params,
                "conv_macs": macs,
                "conv_gmacs": macs / 1e9,
                "conv_flops_if_2flops_per_mac": macs * 2,
                "conv_gflops_if_2flops_per_mac": macs * 2 / 1e9,
                "device": str(device),
                **latency,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "notes": [
                        "MACs count Conv2d layers only.",
                        "FLOPs use the common convention 1 MAC = 2 FLOPs.",
                        "Latency is hardware/framework dependent and should be measured on the target device.",
                    ],
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
