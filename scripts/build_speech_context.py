#!/usr/bin/env python3
"""
Build a speech context sidecar for a SLAM map.

Given a video (with audio) and its SLAM map, this script:
  1. Transcribes the audio with Whisper.
  2. Computes per-keyframe mean embeddings from the map.
  3. Detects scene boundaries via cosine-similarity drops between adjacent keyframes.
  4. Assigns the most recent speech to each keyframe, propagating forward
     within scene boundaries and resetting at scene breaks.
  5. Saves a JSON sidecar: {frame_ind (str) -> context_text (str | null)}.

The sidecar can then be passed to grounding.py via --speech-context to boost
similarity scores for 3D points whose recorded context matches the query.

Usage:
    pip install openai-whisper
    python scripts/build_speech_context.py \
        --video path/to/video.mp4 \
        --map vipe_results/run/vid_slam_map.pt \
        --output vipe_results/run/speech_context.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _get_fps_ffprobe(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    line = result.stdout.strip()
    if "=" in line:
        frac = line.split("=")[1]
        num, den = frac.split("/")
        return float(num) / float(den)
    raise RuntimeError(f"ffprobe could not determine FPS from {video_path!r}; output: {result.stdout!r}")


def get_fps(video_path: str) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return fps
    except ImportError:
        pass
    return _get_fps_ffprobe(video_path)


def build_speech_context(
    video_path: str,
    map_path: str,
    output_path: str,
    whisper_model: str = "base",
    scene_threshold: float = 0.7,
    fps: float | None = None,
) -> None:
    # --- Load map ---
    print(f"Loading map from {map_path} ...")
    data = torch.load(map_path, map_location="cpu")
    frame_inds: list[int] = data["dense_disp_frame_inds"]
    packinfo: torch.Tensor = data["dense_disp_packinfo"]   # (N_kf, V, 2)
    embeddings: torch.Tensor | None = data.get("dense_disp_embeddings")

    n_keyframes = len(frame_inds)
    n_points = len(data["dense_disp_xyz"])
    print(f"  {n_keyframes} keyframes, {n_points} points")

    # --- FPS ---
    if fps is None:
        fps = get_fps(video_path)
    print(f"  Video FPS: {fps:.3f}")

    # --- Per-keyframe mean embeddings (scene change detection) ---
    kf_embeddings: list[torch.Tensor | None] = []
    if embeddings is not None:
        emb_f = embeddings.float()
        for kf_i in range(n_keyframes):
            start = int(packinfo[kf_i, 0, 0].item())
            count = int(packinfo[kf_i, 0, 1].item())
            if count > 0:
                mean_emb = emb_f[start : start + count].mean(0)
                kf_embeddings.append(F.normalize(mean_emb, dim=0))
            else:
                kf_embeddings.append(None)
    else:
        print("  Warning: map has no embeddings — scene change detection disabled.")
        kf_embeddings = [None] * n_keyframes

    # --- Scene boundary detection ---
    boundaries: set[int] = set()
    for i in range(1, n_keyframes):
        a, b = kf_embeddings[i - 1], kf_embeddings[i]
        if a is None or b is None:
            boundaries.add(i)
            continue
        sim = float((a * b).sum().item())   # both already L2-normalised
        if sim < scene_threshold:
            boundaries.add(i)

    print(f"  {len(boundaries)} scene boundaries detected")
    if boundaries:
        sample = sorted(boundaries)[:8]
        suffix = " ..." if len(boundaries) > 8 else ""
        print(f"  Boundary keyframe indices: {sample}{suffix}")

    # --- Whisper transcription ---
    try:
        import whisper as _whisper
    except ImportError:
        print(
            "\nERROR: openai-whisper is not installed.\n"
            "Install it with:  pip install openai-whisper\n"
        )
        sys.exit(1)

    print(f"\nTranscribing audio with Whisper model '{whisper_model}' ...")
    model = _whisper.load_model(whisper_model)
    result = model.transcribe(video_path, word_timestamps=False)
    segments = [
        (float(s["start"]), float(s["end"]), s["text"].strip())
        for s in result["segments"]
        if s["text"].strip()
    ]
    print(f"  {len(segments)} transcript segments")
    for start, end, text in segments[:6]:
        print(f"    [{start:6.1f}s – {end:6.1f}s]  {text}")
    if len(segments) > 6:
        print(f"    ... ({len(segments) - 6} more)")

    # --- Assign and propagate context ---
    # For each keyframe:
    #   - If speech is active at this timestamp, update active context.
    #   - If a scene boundary is hit, reset active context.
    #   - Assign active context (may be None) to this keyframe.
    context: dict[str, str | None] = {}
    active: str | None = None

    for kf_i, frame_ind in enumerate(frame_inds):
        t = frame_ind / fps

        # Check whether speech is happening at this exact timestamp.
        spoken = [
            text for s_start, s_end, text in segments
            if s_start <= t <= s_end
        ]
        if spoken:
            active = " ".join(spoken)

        # Scene break resets propagation.
        if kf_i in boundaries:
            active = None

        context[str(frame_ind)] = active

    covered = sum(1 for v in context.values() if v is not None)
    print(f"\n  Context assigned to {covered}/{n_keyframes} keyframes "
          f"({100 * covered // n_keyframes}%)")

    # --- Save ---
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(context, f, indent=2)
    print(f"\nSaved speech context → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the source video file (must contain audio).",
    )
    parser.add_argument(
        "--map", required=True,
        help="Path to the SLAM map (.pt file).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the speech context JSON sidecar.",
    )
    parser.add_argument(
        "--whisper-model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base). Larger = more accurate but slower.",
    )
    parser.add_argument(
        "--scene-threshold", type=float, default=0.7,
        help=(
            "Cosine similarity below which adjacent keyframes are considered a "
            "scene change and context propagation resets (default: 0.7). "
            "Lower → fewer resets; higher → more resets."
        ),
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Video FPS. Auto-detected from the file if not provided.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not Path(args.video).exists():
        print(f"Error: video not found: {args.video}")
        sys.exit(1)
    if not Path(args.map).exists():
        print(f"Error: map not found: {args.map}")
        sys.exit(1)
    build_speech_context(
        video_path=args.video,
        map_path=args.map,
        output_path=args.output,
        whisper_model=args.whisper_model,
        scene_threshold=args.scene_threshold,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
