#!/usr/bin/env python3
"""
Overlay a second video's camera trail on an existing SLAM map.

Given two SLAM maps (map1 = reference, map2 = new video), this script:
  1. Computes per-keyframe mean embeddings for both maps.
  2. Finds matching keyframe pairs via mutual nearest-neighbour cosine similarity.
  3. Estimates a rigid alignment (R, t) from the matched camera positions.
  4. Renders in Rerun: map1 RGB point cloud + map2 camera trail (transformed).

Usage:
    python scripts/localize_video.py \
        --map1 vipe_results/room_15fps/vipe/vid_slam_map.pt \
        --map2 vipe_results/room2_15fps/vipe/vid_slam_map.pt \
        --output localize.rrd
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def procrustes_align(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid alignment (no scale): find R, t so that dst ≈ R @ src + t.

    Args:
        src: (N, 3) source positions (map2 keyframe positions)
        dst: (N, 3) target positions (matched map1 keyframe positions)
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    H = (src - mu_src).T @ (dst - mu_dst)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:   # fix reflection
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def transform_positions(positions: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ positions.T).T + t


# ---------------------------------------------------------------------------
# Map utilities
# ---------------------------------------------------------------------------

def load_map(path: Path) -> dict:
    print(f"Loading {path} ...")
    data = torch.load(path, map_location="cpu")
    n_pts = len(data["dense_disp_xyz"])
    n_kf  = len(data["dense_disp_frame_inds"])
    print(f"  {n_kf} keyframes, {n_pts} points")
    if data.get("keyframe_poses") is not None:
        print(f"  keyframe_poses: {data['keyframe_poses'].shape}")
    else:
        print("  keyframe_poses: not saved (will use point centroids)")
    return data


def get_keyframe_positions(data: dict) -> np.ndarray:
    """(N_kf, 3) camera positions. Uses saved poses if available, else centroids."""
    if data.get("keyframe_poses") is not None:
        # SE3 format: [tx, ty, tz, qx, qy, qz, qw] — translation is first 3
        return data["keyframe_poses"][:, :3].numpy()

    # Fallback: centroid of each keyframe's 3D points
    xyz      = data["dense_disp_xyz"].numpy()
    packinfo = data["dense_disp_packinfo"]
    n_kf     = len(data["dense_disp_frame_inds"])
    positions = []
    for kf_i in range(n_kf):
        start = int(packinfo[kf_i, 0, 0].item())
        count = int(packinfo[kf_i, 0, 1].item())
        if count > 0:
            positions.append(xyz[start : start + count].mean(0))
        else:
            positions.append(np.zeros(3))
    return np.array(positions)


def get_keyframe_embeddings(data: dict) -> torch.Tensor | None:
    """(N_kf, C) mean embedding per keyframe, L2-normalised. None if no embeddings."""
    embeddings = data.get("dense_disp_embeddings")
    if embeddings is None:
        return None
    packinfo = data["dense_disp_packinfo"]
    n_kf     = len(data["dense_disp_frame_inds"])
    emb_f    = embeddings.float()
    kf_embs  = []
    for kf_i in range(n_kf):
        start = int(packinfo[kf_i, 0, 0].item())
        count = int(packinfo[kf_i, 0, 1].item())
        if count > 0:
            kf_embs.append(emb_f[start : start + count].mean(0))
        else:
            kf_embs.append(torch.zeros(emb_f.shape[-1]))
    stacked = torch.stack(kf_embs, 0)
    return F.normalize(stacked, dim=-1)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_keyframes(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    min_similarity: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest-neighbour matching between two sets of keyframe embeddings.

    Returns:
        idx1: matched keyframe indices in map1
        idx2: matched keyframe indices in map2
    """
    sim = emb2 @ emb1.T    # (N2, N1)

    nn2_in_1 = sim.argmax(dim=1)   # for each kf in map2, best match in map1
    nn1_in_2 = sim.argmax(dim=0)   # for each kf in map1, best match in map2

    idx2_list, idx1_list = [], []
    for i2, i1 in enumerate(nn2_in_1.tolist()):
        if nn1_in_2[i1].item() == i2:             # mutual
            if sim[i2, i1].item() >= min_similarity:
                idx2_list.append(i2)
                idx1_list.append(i1)

    return np.array(idx1_list), np.array(idx2_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def localize(
    map1_path: Path,
    map2_path: Path,
    output_path: Path,
    min_similarity: float = 0.5,
    trail_color: tuple[int, int, int] = (255, 165, 0),
) -> None:

    data1 = load_map(map1_path)
    data2 = load_map(map2_path)

    pos1 = get_keyframe_positions(data1)   # (N1, 3)
    pos2 = get_keyframe_positions(data2)   # (N2, 3)
    print(f"\nMap1: {len(pos1)} keyframe positions")
    print(f"Map2: {len(pos2)} keyframe positions")

    emb1 = get_keyframe_embeddings(data1)
    emb2 = get_keyframe_embeddings(data2)

    if emb1 is None or emb2 is None:
        print("WARNING: one or both maps have no embeddings — "
              "skipping alignment, rendering map2 trail in its own frame.")
        R, t = np.eye(3), np.zeros(3)
        n_matches = 0
    else:
        idx1, idx2 = match_keyframes(emb1, emb2, min_similarity=min_similarity)
        n_matches = len(idx1)
        print(f"\nMutual NN matches (similarity ≥ {min_similarity}): {n_matches}")

        if n_matches < 3:
            print("WARNING: fewer than 3 matches — cannot estimate alignment reliably.")
            print("  Try lowering --min-similarity or use a map with more overlap.")
            R, t = np.eye(3), np.zeros(3)
        else:
            R, t = procrustes_align(pos2[idx2], pos1[idx1])
            residuals = np.linalg.norm(
                transform_positions(pos2[idx2], R, t) - pos1[idx1], axis=1
            )
            print(f"Alignment residuals: mean={residuals.mean():.3f}m, "
                  f"max={residuals.max():.3f}m")

    # Transform map2 positions into map1 frame
    trail = transform_positions(pos2, R, t)

    # --- Rerun ---
    rr.init("SLAM Localization")
    rr.save(str(output_path))

    # Map1: RGB point cloud
    xyz1 = data1["dense_disp_xyz"].numpy()
    rgb1 = (data1["dense_disp_rgb"].numpy() * 255).astype(np.uint8)
    rr.log("world/map1/points", rr.Points3D(positions=xyz1, colors=rgb1, radii=0.008))

    # Map2: camera trail as a polyline + individual position markers
    rr.log("world/map2/trail",
           rr.LineStrips3D([trail], colors=[trail_color]))
    rr.log("world/map2/positions",
           rr.Points3D(positions=trail, colors=[trail_color], radii=0.03))

    # Highlight matched keyframes on both maps
    if n_matches >= 3:
        rr.log("world/matches/map1",
               rr.Points3D(positions=pos1[idx1], colors=[0, 255, 0], radii=0.05))
        rr.log("world/matches/map2",
               rr.Points3D(positions=transform_positions(pos2[idx2], R, t),
                           colors=[0, 200, 255], radii=0.05))

    # Coordinate frame at origin
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
    parser.add_argument("--map1", required=True, type=Path,
                        help="Reference SLAM map (.pt) — point cloud shown in full colour.")
    parser.add_argument("--map2", required=True, type=Path,
                        help="Second video SLAM map (.pt) — camera trail overlaid on map1.")
    parser.add_argument("--output", type=Path, default=Path("localize.rrd"),
                        help="Output .rrd file (default: localize.rrd).")
    parser.add_argument("--min-similarity", type=float, default=0.5,
                        help="Minimum cosine similarity for a keyframe match to be used "
                             "in alignment (default: 0.5). Lower if too few matches.")
    parser.add_argument("--trail-color", type=str, default="255,165,0",
                        help="RGB colour for the camera trail, comma-separated "
                             "(default: 255,165,0 = orange).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for p in [args.map1, args.map2]:
        if not p.exists():
            print(f"Error: {p} not found")
            sys.exit(1)
    color = tuple(int(x) for x in args.trail_color.split(","))
    localize(
        map1_path=args.map1,
        map2_path=args.map2,
        output_path=args.output,
        min_similarity=args.min_similarity,
        trail_color=color,
    )


if __name__ == "__main__":
    main()
