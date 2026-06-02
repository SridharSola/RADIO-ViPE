#!/usr/bin/env python3
"""
Lightweight camera localization against an existing SLAM map.

No SLAM on the new video. Pipeline per frame:
  new frame → RADIO backbone → mean-pool → PCA compress → nearest keyframe → position

Requires only the RADIO model (already loaded for grounding).
No depth estimation, no bundle adjustment, no loop closure.

Usage:
    python scripts/localize_lightweight.py \
        --map  vipe_results/room/vipe/vid_slam_map.pt \
        --pca  vipe_results/room/vipe/pca_basis.pt \
        --video room2.mp4 \
        --output localize.rrd
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F

from vipe.priors.embedding.radseg_encoder import RADSegEncoder


# ---------------------------------------------------------------------------
# PCA helpers (reuse the same basis saved by the SLAM run)
# ---------------------------------------------------------------------------

def load_pca_basis(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load mean and basis from pca_basis.pt. Returns (mean, basis) on device."""
    data = torch.load(path, map_location="cpu")
    # Basis can be stored at multiple levels of nesting; search for mean + basis/components.
    def _find(d):
        if isinstance(d, dict):
            if "mean" in d and ("basis" in d or "components" in d):
                mean = d["mean"]
                basis = d.get("basis", d.get("components"))
                return mean, basis
            for v in d.values():
                result = _find(v)
                if result is not None:
                    return result
        return None
    result = _find(data)
    if result is None:
        raise ValueError(f"Could not find mean/basis in {path}")
    mean, basis = result
    return mean.to(device, dtype=torch.float32), basis.to(device, dtype=torch.float32)


def pca_compress(x: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project x (*, D) into PCA space: (x - mean) @ basis → (*, K)."""
    return (x - mean) @ basis


# ---------------------------------------------------------------------------
# Per-keyframe data from map
# ---------------------------------------------------------------------------

def get_keyframe_positions(data: dict) -> np.ndarray:
    """(N_kf, 3) camera positions. Saved poses if available, else point centroids."""
    if data.get("keyframe_poses") is not None:
        return data["keyframe_poses"][:, :3].float().numpy()
    xyz      = data["dense_disp_xyz"].numpy()
    packinfo = data["dense_disp_packinfo"]
    n_kf     = len(data["dense_disp_frame_inds"])
    positions = []
    for i in range(n_kf):
        start = int(packinfo[i, 0, 0].item())
        count = int(packinfo[i, 0, 1].item())
        positions.append(xyz[start : start + count].mean(0) if count > 0 else np.zeros(3))
    return np.array(positions, dtype=np.float32)


def get_keyframe_embeddings(data: dict) -> torch.Tensor | None:
    """(N_kf, C) mean embedding per keyframe, L2-normalised. None if no embeddings."""
    embs = data.get("dense_disp_embeddings")
    if embs is None:
        return None
    packinfo = data["dense_disp_packinfo"]
    n_kf     = len(data["dense_disp_frame_inds"])
    emb_f    = embs.float()
    out = []
    for i in range(n_kf):
        start = int(packinfo[i, 0, 0].item())
        count = int(packinfo[i, 0, 1].item())
        out.append(emb_f[start : start + count].mean(0) if count > 0
                   else torch.zeros(emb_f.shape[-1]))
    return F.normalize(torch.stack(out), dim=-1)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def read_video_frames(video_path: str, sample_every: int):
    """Yield (frame_idx, rgb_tensor) for every sample_every-th frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.1f} fps → "
          f"processing every {sample_every}th frame "
          f"({total // sample_every} frames)")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield idx, rgb
        idx += 1
    cap.release()


def preprocess_frame(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """HxWx3 uint8 → 1x3xHxW float32 [0,1] on device."""
    t = torch.from_numpy(rgb).float() / 255.0   # (H, W, 3)
    t = t.permute(2, 0, 1).unsqueeze(0)          # (1, 3, H, W)
    return t.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def localize(
    map_path: Path,
    pca_path: Path,
    video_path: str,
    output_path: Path,
    sample_every: int = 5,
    smooth_window: int = 7,
    device: str = "cpu",
    model_version: str = "c-radio_v3-b",
    lang_model: str = "siglip2",
    trail_color: tuple[int, int, int] = (255, 165, 0),
) -> None:

    torch_device = torch.device(device)

    # --- Load map ---
    print(f"Loading map: {map_path}")
    data = torch.load(map_path, map_location="cpu")
    kf_positions = get_keyframe_positions(data)          # (N_kf, 3)
    kf_embeddings = get_keyframe_embeddings(data)        # (N_kf, C_pca) or None
    if kf_embeddings is None:
        print("Error: map has no embeddings — cannot localise.")
        sys.exit(1)
    kf_embeddings = kf_embeddings.to(torch_device)
    print(f"  {len(kf_positions)} keyframes, embedding dim {kf_embeddings.shape[-1]}")

    # --- Load PCA basis ---
    print(f"Loading PCA basis: {pca_path}")
    pca_mean, pca_basis = load_pca_basis(pca_path, torch_device)
    print(f"  PCA: {pca_basis.shape[0]}D → {pca_basis.shape[1]}D")

    # --- Load RADIO encoder ---
    print("Loading RADIO encoder (this takes 1-2 min) ...")
    t0 = time.time()
    encoder = RADSegEncoder(
        model_version=model_version,
        lang_model=lang_model,
        sam_refinement=False,
        predict=False,
        device=torch_device,
    )
    print(f"  Encoder ready in {time.time() - t0:.0f}s")

    # --- Count frames to process ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n_to_process = max(1, total_frames // sample_every)

    # --- Process video frames ---
    raw_positions = []
    t_start = time.time()
    print(f"\nLocalising {n_to_process} frames ...")

    for frame_idx, rgb in read_video_frames(video_path, sample_every):
        img = preprocess_frame(rgb, torch_device)

        with torch.inference_mode():
            feat_map = encoder.encode_image_to_feat_map(img)
            feat_map = encoder.align_spatial_features_with_language(feat_map, onehot=False)
            global_desc = feat_map.mean(dim=[2, 3]).squeeze(0)
            compressed = F.normalize(pca_compress(global_desc, pca_mean, pca_basis), dim=0)

        sims = (kf_embeddings @ compressed).cpu()
        best = int(sims.argmax().item())
        raw_positions.append(kf_positions[best])

        n_done = len(raw_positions)
        if n_done % 5 == 0 or n_done == n_to_process:
            elapsed = time.time() - t_start
            fps_proc = n_done / elapsed if elapsed > 0 else 0
            eta = (n_to_process - n_done) / fps_proc if fps_proc > 0 else 0
            print(f"  [{n_done}/{n_to_process}]  "
                  f"{fps_proc:.1f} frames/s  "
                  f"ETA {eta:.0f}s", end="\r", flush=True)

    print(f"\n  Done in {time.time() - t_start:.0f}s")

    if not raw_positions:
        print("No frames processed.")
        sys.exit(1)

    trail_raw = np.array(raw_positions)   # (T, 3)

    # --- Smooth trajectory ---
    half = smooth_window // 2
    trail = np.array([
        np.median(trail_raw[max(0, i - half) : i + half + 1], axis=0)
        for i in range(len(trail_raw))
    ])

    print(f"\nTrail: {len(trail)} positions")

    # --- Render ---
    rr.init("SLAM Localization")
    rr.save(str(output_path))

    xyz = data["dense_disp_xyz"].numpy()
    rgb = (data["dense_disp_rgb"].numpy() * 255).astype(np.uint8)
    rr.log("world/map/points",
           rr.Points3D(positions=xyz, colors=rgb, radii=0.008))

    rr.log("world/trail/line",
           rr.LineStrips3D([trail], colors=[trail_color]))
    rr.log("world/trail/positions",
           rr.Points3D(positions=trail, colors=[trail_color], radii=0.025))

    rr.log("world/axes", rr.LineStrips3D([
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [0, 0, 1]],
    ]))

    print(f"\nSaved → {output_path}")
    print(f"Download and open with:  rerun {output_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--map",   required=True, type=Path,
                        help="Existing SLAM map (.pt).")
    parser.add_argument("--pca",   required=True, type=Path,
                        help="PCA basis saved alongside the map (pca_basis.pt).")
    parser.add_argument("--video", required=True,
                        help="New video to localise (any fps — will be sampled).")
    parser.add_argument("--output", type=Path, default=Path("localize.rrd"),
                        help="Output .rrd file (default: localize.rrd).")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Process every Nth frame (default: 5). "
                             "Higher = faster but coarser trail.")
    parser.add_argument("--smooth-window", type=int, default=7,
                        help="Median filter window for trail smoothing (default: 7).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Device (default: cpu).")
    parser.add_argument("--model-version", default="c-radio_v3-b",
                        help="RADIO model variant (default: c-radio_v3-b).")
    parser.add_argument("--trail-color", default="255,165,0",
                        help="RGB trail colour, comma-separated (default: orange).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for p, name in [(args.map, "--map"), (args.pca, "--pca")]:
        if not p.exists():
            print(f"Error: {name} file not found: {p}")
            sys.exit(1)
    color = tuple(int(x) for x in args.trail_color.split(","))
    localize(
        map_path=args.map,
        pca_path=args.pca,
        video_path=args.video,
        output_path=args.output,
        sample_every=args.sample_every,
        smooth_window=args.smooth_window,
        device=args.device,
        trail_color=color,
    )


if __name__ == "__main__":
    main()
