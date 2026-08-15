from __future__ import annotations

import argparse
import json
import sys

from bci.config import load_config
from bci.evaluation.runner import bootstrap_dataset, run_evaluation, run_lsl_replay_then_evaluate
from bci.registry import get_dataset_adapter
from bci.sources.lsl import LSLEEGSource
from bci.sources.replay import MOABBReplayPublisher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bci")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["bootstrap", "evaluate", "replay", "run", "inspect"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config.project.data_dir.mkdir(parents=True, exist_ok=True)
    config.project.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.project.log_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "bootstrap":
        meta = bootstrap_dataset(config)
        print(json.dumps(meta, indent=2))
        return 0
    if args.command == "evaluate":
        metrics = run_evaluation(config, "eval")
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "replay":
        adapter = get_dataset_adapter(config)
        adapter.ensure_available()
        publisher = MOABBReplayPublisher(config, adapter)
        publisher.start()
        print(f"Streaming {config.dataset.name} on LSL stream {config.source.replay.stream_name}. Press Ctrl+C to stop.")
        try:
            import time

            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            publisher.stop()
        return 0
    if args.command == "inspect":
        source = LSLEEGSource(config)
        meta = source.connect()
        print(json.dumps(meta.__dict__, indent=2))
        source.close()
        return 0
    if args.command == "run":
        metrics = run_lsl_replay_then_evaluate(config)
        print(json.dumps(metrics, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
