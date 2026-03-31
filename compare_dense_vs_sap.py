"""Compare dense vs SAP sparse attention: timing and quality metrics."""
import argparse
import json
import os
import time

import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

import sys
sys.path.insert(0, "svg/kernels/build")

from svg.models.wan.inference import replace_wan_attention
from svg.utils.seed import seed_everything
from svg.logger import logger
from copy import deepcopy
import math


def compute_psnr(vid_a, vid_b):
    """Compute PSNR between two video tensors [T, H, W, C] in 0-255 uint8."""
    a = vid_a.float()
    b = vid_b.float()
    mse = ((a - b) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim_frame(a, b):
    """Simple SSIM for single frame (H,W,C) float tensors in [0,255]."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a_sq = ((a - mu_a) ** 2).mean()
    sigma_b_sq = ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    ssim = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / \
           ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a_sq + sigma_b_sq + C2))
    return ssim.item()


def compute_ssim(vid_a, vid_b):
    """Average SSIM across frames."""
    a = vid_a.float()
    b = vid_b.float()
    ssims = [compute_ssim_frame(a[t], b[t]) for t in range(a.shape[0])]
    return np.mean(ssims)


def frames_to_tensor(frames):
    """Convert list of PIL images to (T, H, W, C) uint8 tensor."""
    return torch.stack([torch.from_numpy(np.array(f)) for f in frames])


def run_inference(pipe, args, pattern, sap_args=None):
    """Run inference and return (frames_list, elapsed_seconds)."""
    seed_everything(args.seed)

    torch.cuda.synchronize()
    t0 = time.time()

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=args.num_inference_steps,
    ).frames[0]

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    return output, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompts_file", default="configs/prompts.txt")
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/comparison-1.3b-480p")
    # SAP params
    parser.add_argument("--num_q_centroids", type=int, default=300)
    parser.add_argument("--num_k_centroids", type=int, default=1000)
    parser.add_argument("--top_p_kmeans", type=float, default=0.9)
    parser.add_argument("--min_kc_ratio", type=float, default=0.10)
    parser.add_argument("--kmeans_iter_init", type=int, default=50)
    parser.add_argument("--kmeans_iter_step", type=int, default=2)
    parser.add_argument("--first_times_fp", type=float, default=0.2)
    parser.add_argument("--first_layers_fp", type=float, default=0.03)
    args = parser.parse_args()

    args.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts
    with open(args.prompts_file) as f:
        prompts = [l.strip() for l in f if l.strip()]
    prompts = prompts[:args.num_prompts]

    # Load model
    torch.backends.cuda.preferred_linalg_library(backend="magma")
    model_id = args.model_id
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 3.0  # 3.0 for 480P
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True,
        num_train_timesteps=1000, flow_shift=flow_shift
    )
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to("cuda")

    config = pipe.transformer.config
    ref_scheduler = deepcopy(pipe.scheduler)
    ref_scheduler.set_timesteps(args.num_inference_steps)

    num_fp_timesteps = math.floor(args.first_times_fp * args.num_inference_steps)
    num_fp_layers = math.floor(args.first_layers_fp * config.num_layers)
    first_times_fp_val = ref_scheduler.timesteps[num_fp_timesteps - 1] - 1 if num_fp_timesteps > 0 else 1001
    first_layers_fp_val = num_fp_layers

    results = []

    for pi, prompt in enumerate(prompts):
        args.prompt = prompt
        padded = f"{pi:03d}"

        # ===== DENSE =====
        logger.info(f"[Dense] Prompt {pi}: {prompt[:60]}...")

        # Reset attention to dense (reload processors)
        pipe_dense = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        pipe_dense.scheduler = deepcopy(scheduler)
        pipe_dense.to("cuda")

        dense_frames, dense_time = run_inference(pipe_dense, args, "dense")
        dense_path = f"{args.output_dir}/dense_p{padded}_s{args.seed}.mp4"
        export_to_video(dense_frames, dense_path, fps=16)

        del pipe_dense
        torch.cuda.empty_cache()

        # ===== SAP =====
        logger.info(f"[SAP] Prompt {pi}: {prompt[:60]}...")

        pipe_sap = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        pipe_sap.scheduler = deepcopy(scheduler)
        pipe_sap.to("cuda")

        replace_wan_attention(
            pipe_sap, args.height, args.width, args.num_frames,
            first_layers_fp=first_layers_fp_val,
            first_times_fp=first_times_fp_val,
            pattern="SAP",
            num_q_centroids=args.num_q_centroids,
            num_k_centroids=args.num_k_centroids,
            top_p_kmeans=args.top_p_kmeans,
            min_kc_ratio=args.min_kc_ratio,
            kmeans_iter_init=args.kmeans_iter_init,
            kmeans_iter_step=args.kmeans_iter_step,
        )

        sap_frames, sap_time = run_inference(pipe_sap, args, "SAP")
        sap_path = f"{args.output_dir}/sap_p{padded}_s{args.seed}.mp4"
        export_to_video(sap_frames, sap_path, fps=16)

        del pipe_sap
        torch.cuda.empty_cache()

        # ===== Quality comparison =====
        dense_tensor = frames_to_tensor(dense_frames)
        sap_tensor = frames_to_tensor(sap_frames)

        psnr = compute_psnr(dense_tensor, sap_tensor)
        ssim = compute_ssim(dense_tensor, sap_tensor)

        speedup = dense_time / sap_time if sap_time > 0 else 0

        result = {
            "prompt_idx": pi,
            "prompt": prompt,
            "seed": args.seed,
            "dense_time_s": round(dense_time, 2),
            "sap_time_s": round(sap_time, 2),
            "speedup": round(speedup, 3),
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim, 4),
        }
        results.append(result)

        logger.info(
            f"  Dense: {dense_time:.1f}s | SAP: {sap_time:.1f}s | "
            f"Speedup: {speedup:.2f}x | PSNR: {psnr:.1f}dB | SSIM: {ssim:.4f}"
        )

    # Save results
    results_path = f"{args.output_dir}/comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    avg_dense = np.mean([r["dense_time_s"] for r in results])
    avg_sap = np.mean([r["sap_time_s"] for r in results])
    avg_speedup = np.mean([r["speedup"] for r in results])
    avg_psnr = np.mean([r["psnr_db"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])

    print("\n" + "=" * 70)
    print(f"COMPARISON SUMMARY: Dense vs SAP (Wan2.1-1.3B, 480p, {args.num_frames}f)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Dense':<15} {'SAP':<15} {'Ratio':<15}")
    print("-" * 70)
    print(f"{'Avg time (s)':<25} {avg_dense:<15.1f} {avg_sap:<15.1f} {avg_speedup:<15.2f}x")
    print(f"{'Avg PSNR (dB)':<25} {'(ref)':<15} {avg_psnr:<15.1f}")
    print(f"{'Avg SSIM':<25} {'(ref)':<15} {avg_ssim:<15.4f}")
    print("=" * 70)
    print(f"Results saved to {results_path}")
