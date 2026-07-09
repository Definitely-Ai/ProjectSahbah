from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_pose_csv, write_csvs
from .metrics import compute_jitter, compare_sources, summarize_jitter
from .report import generate_report
from .sample_data import write_sample_pose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pose-jitter",
        description="Analyze and compare shoulder-normalized pose jitter across coordinate domains.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── analyze ────────────────────────────────────────────────────────────
    analyze = subparsers.add_parser("analyze", help="Analyze a pose CSV.")
    analyze.add_argument("csv", type=Path, help="Long-form pose CSV.")
    analyze.add_argument("--out", type=Path, default=Path("reports") / "pose_jitter", help="Output directory.")
    analyze.add_argument(
        "--shoulder-mode",
        choices=["trial_median", "frame", "pair"],
        default="trial_median",
        help="Shoulder denominator to use for normalized jitter.",
    )
    analyze.add_argument("--left-shoulder", default="left_shoulder", help="Left shoulder landmark name.")
    analyze.add_argument("--right-shoulder", default="right_shoulder", help="Right shoulder landmark name.")
    analyze.add_argument("--title", default="Pose Jitter Lab Report", help="Report title.")
    analyze.add_argument("--no-report", action="store_true", help="Skip HTML report generation.")
    analyze.add_argument("--export-figures", action="store_true", help="Export individual PNG figures for thesis.")
    analyze.add_argument("--no-stats", action="store_true", help="Skip statistical analysis in the report.")
    analyze.set_defaults(func=run_analyze)

    # ── batch ──────────────────────────────────────────────────────────────
    batch = subparsers.add_parser("batch", help="Batch analyze multiple pose CSVs.")
    batch.add_argument("csvs", type=Path, nargs="+", help="One or more long-form pose CSVs.")
    batch.add_argument("--out", type=Path, default=Path("reports") / "batch", help="Output directory.")
    batch.add_argument(
        "--shoulder-mode",
        choices=["trial_median", "frame", "pair"],
        default="trial_median",
        help="Shoulder denominator to use for normalized jitter.",
    )
    batch.add_argument("--left-shoulder", default="left_shoulder", help="Left shoulder landmark name.")
    batch.add_argument("--right-shoulder", default="right_shoulder", help="Right shoulder landmark name.")
    batch.add_argument("--title", default="Batch Pose Jitter Analysis", help="Report title.")
    batch.add_argument("--source-from-filename", action="store_true",
                       help="Use the CSV filename as the source label.")
    batch.add_argument("--trial-from-filename", action="store_true", default=True,
                       help="Use the CSV filename as the trial label (default: True).")
    batch.add_argument("--export-figures", action="store_true", help="Export individual PNG figures.")
    batch.add_argument("--no-stats", action="store_true", help="Skip statistical analysis.")
    batch.add_argument("--phase-split", action="store_true",
                       help="Split each trial into movement phases.")
    batch.add_argument("--n-phases", type=int, default=3, help="Number of movement phases.")
    batch.add_argument("--phase-names", nargs="+", default=None,
                       help="Custom phase names (e.g., reach hold return).")
    batch.set_defaults(func=run_batch_cli)

    # ── demo ───────────────────────────────────────────────────────────────
    demo = subparsers.add_parser("demo", help="Generate demo data and analyze it.")
    demo.add_argument("--out", type=Path, default=Path("reports") / "demo", help="Output directory.")
    demo.add_argument("--data", type=Path, default=Path("data") / "example_pose.csv", help="Demo CSV path.")
    demo.add_argument("--frames", type=int, default=180, help="Number of demo frames.")
    demo.set_defaults(func=run_demo)

    # ── cameras ────────────────────────────────────────────────────────────
    cameras = subparsers.add_parser("cameras", help="List available webcam indexes.")
    cameras.add_argument("--max-index", type=int, default=8, help="Number of camera indexes to probe.")
    cameras.add_argument("--backend", choices=["dshow", "msmf", "any"], default="dshow", help="OpenCV capture backend.")
    cameras.set_defaults(func=run_cameras)

    # ── live ───────────────────────────────────────────────────────────────
    live = subparsers.add_parser("live", help="Open a real-time MediaPipe pose landmark and jitter viewer.")
    live.add_argument("--camera", type=int, default=0, help="Webcam index.")
    live.add_argument("--model", type=Path, default=Path("models") / "pose_landmarker_lite.task", help="MediaPipe pose .task model path.")
    live.add_argument("--record", type=Path, default=None, help="CSV output path. Defaults to data/<trial>_pose.csv.")
    live.add_argument("--auto-record", action="store_true", help="Start recording immediately.")
    live.add_argument("--trial", default=None, help="Trial name stored in the CSV.")
    live.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    live.add_argument("--height", type=int, default=720, help="Requested camera height.")
    live.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    live.add_argument("--backend", choices=["dshow", "msmf", "any"], default="dshow", help="OpenCV capture backend.")
    live.add_argument("--visibility-min", type=float, default=0.35, help="Minimum landmark visibility for drawing and live metrics.")
    live.add_argument("--presence-min", type=float, default=0.35, help="Minimum landmark presence confidence for live metrics.")
    live.add_argument("--jitter-window", type=int, default=45, help="Rolling jitter window in frames.")
    live.add_argument("--include-z", action="store_true", help="Include MediaPipe normalized z in live jitter math.")
    live.add_argument("--mirror-display", action="store_true", help="Mirror the displayed and processed camera frame.")
    live.add_argument("--max-frames", type=int, default=None, help="Stop after N frames, mostly for verification.")
    live.add_argument("--headless", action="store_true", help="Run without opening a window, mostly for verification.")
    live.add_argument("--save-preview", type=Path, default=None, help="Save the first processed frame to an image path.")
    live.set_defaults(func=run_live_cli)

    # ── stereo-preview ─────────────────────────────────────────────────────
    stereo = subparsers.add_parser("stereo-preview", help="Open two webcams side by side with pose landmarks.")
    stereo.add_argument("--left-camera", type=int, required=True, help="Left webcam index.")
    stereo.add_argument("--right-camera", type=int, required=True, help="Right webcam index.")
    stereo.add_argument("--model", type=Path, default=Path("models") / "pose_landmarker_lite.task", help="MediaPipe pose .task model path.")
    stereo.add_argument("--backend", choices=["dshow", "msmf", "any"], default="dshow", help="OpenCV capture backend.")
    stereo.add_argument("--max-frames", type=int, default=None, help="Stop after N frames, mostly for verification.")
    stereo.set_defaults(func=run_stereo_preview_cli)

    return parser


def run_analyze(args: argparse.Namespace) -> int:
    pose = load_pose_csv(args.csv)
    jitter, shoulder_widths = compute_jitter(
        pose,
        shoulder_mode=args.shoulder_mode,
        left_shoulder=args.left_shoulder,
        right_shoulder=args.right_shoulder,
    )
    summary = summarize_jitter(jitter)
    comparison = compare_sources(summary)

    written = write_csvs(
        {
            "jitter_frames": jitter,
            "jitter_summary": summary,
            "comparison": comparison,
            "shoulder_widths": shoulder_widths,
        },
        args.out,
    )
    report_path = None
    if not args.no_report:
        report_path = generate_report(
            pose=pose,
            jitter=jitter,
            summary=summary,
            shoulder_widths=shoulder_widths,
            output_path=args.out / "report.html",
            title=args.title,
            shoulder_mode=args.shoulder_mode,
        )

    if args.export_figures:
        from .aggregate import _export_figures
        _export_figures(jitter, summary, shoulder_widths, args.out / "figures")
        print(f"Exported figures to: {args.out / 'figures'}")

    print(f"Pose rows: {len(pose)}")
    print(f"Frame-to-frame jitter rows: {len(jitter)}")
    print(f"Summary rows: {len(summary)}")
    for name, path in written.items():
        print(f"Wrote {name}: {path}")
    if report_path:
        print(f"Wrote report: {report_path}")

    if not summary.empty:
        top = summary.sort_values("normalized_mean", ascending=False).iloc[0]
        print(
            "Highest mean normalized jitter: "
            f"{top['source']} / {top['joint']} = {top['normalized_mean']:.6g}"
        )
    return 0


def run_batch_cli(args: argparse.Namespace) -> int:
    from .aggregate import run_batch

    written = run_batch(
        paths=args.csvs,
        output_dir=args.out,
        shoulder_mode=args.shoulder_mode,
        left_shoulder=args.left_shoulder,
        right_shoulder=args.right_shoulder,
        source_from_filename=args.source_from_filename,
        trial_from_filename=args.trial_from_filename,
        phase_split=args.phase_split,
        n_phases=args.n_phases,
        phase_names=args.phase_names,
        export_figures=args.export_figures,
        include_stats=not args.no_stats,
        title=args.title,
    )

    for name, path in written.items():
        print(f"Wrote {name}: {path}")

    return 0


def run_demo(args: argparse.Namespace) -> int:
    data_path = write_sample_pose(args.data, frames=args.frames)
    analyze_args = argparse.Namespace(
        csv=data_path,
        out=args.out,
        shoulder_mode="trial_median",
        left_shoulder="left_shoulder",
        right_shoulder="right_shoulder",
        title="Pose Jitter Lab Demo Report",
        no_report=False,
        export_figures=False,
        no_stats=False,
    )
    print(f"Wrote demo data: {data_path}")
    return run_analyze(analyze_args)


def run_cameras(args: argparse.Namespace) -> int:
    from .live import run_camera_list

    return run_camera_list(max_index=args.max_index, backend=args.backend)


def run_live_cli(args: argparse.Namespace) -> int:
    from .live import LiveConfig, run_live

    return run_live(
        LiveConfig(
            camera=args.camera,
            model=args.model,
            record=args.record,
            auto_record=args.auto_record,
            trial=args.trial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend=args.backend,
            visibility_min=args.visibility_min,
            presence_min=args.presence_min,
            jitter_window=args.jitter_window,
            include_z=args.include_z,
            max_frames=args.max_frames,
            headless=args.headless,
            mirror_display=args.mirror_display,
            save_preview=args.save_preview,
        )
    )


def run_stereo_preview_cli(args: argparse.Namespace) -> int:
    from .live import run_stereo_preview

    return run_stereo_preview(
        left_camera=args.left_camera,
        right_camera=args.right_camera,
        model=args.model,
        backend=args.backend,
        max_frames=args.max_frames,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
