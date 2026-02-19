"""Browse a LeRobot dataset in the FiftyOne App."""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse a LeRobot dataset in FiftyOne")
    parser.add_argument("repo_ids", help="Comma-separated repo IDs (e.g. lerobot/pusht,lerobot/aloha_sim)")
    parser.add_argument("--subsample-rate", type=int, default=10)
    parser.add_argument("--output-dir", default="/tmp/yavla_fiftyone")
    parser.add_argument("--port", type=int, default=5151)
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--no-app", action="store_true", help="Load dataset without launching the app")
    args = parser.parse_args()

    import fiftyone as fo  # type: ignore[import-untyped]

    from yavla.visualization import load_lerobot_to_fiftyone

    datasets = []
    for repo_id in args.repo_ids.split(","):
        datasets.append(
            load_lerobot_to_fiftyone(
                repo_id,
                subsample_rate=args.subsample_rate,
                output_dir=args.output_dir,
                persistent=args.persistent,
            )
        )

    for ds in datasets:
        print(f"\n  Dataset: {ds.name}  |  Samples: {len(ds)}")

    if args.no_app:
        return

    try:
        session = fo.launch_app(datasets[0], port=args.port)
        print(f"\nFiftyOne running at http://localhost:{args.port}")
        print(f"Loaded {len(datasets)} dataset(s). Switch in the app via the dataset selector.")
        session.wait()
    except Exception as exc:
        print(f"\nFiftyOne app failed to launch: {exc}")
        print("This is a known issue with fiftyone on Python 3.13.")
        print("Datasets were loaded successfully. Use --persistent and access via a Python 3.12 env.")


if __name__ == "__main__":
    main()
