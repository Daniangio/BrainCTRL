from __future__ import annotations

import argparse
import json
import sys

from bci.config import load_config
from bci.evaluation.runner import bootstrap_dataset, run_evaluation
from bci.experiment.factory import build_realtime_experiment
from bci.registry import get_dataset_adapter
from bci.sources.lsl import LSLEEGSource
from bci.sources.replay import MOABBReplayPublisher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bci")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["bootstrap", "evaluate", "replay", "run", "inspect", "experiment", "gui"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--subject", type=int)
        p.add_argument("--model")
        p.add_argument("--max-trials", type=int)
        gui_group = p.add_mutually_exclusive_group()
        gui_group.add_argument("--gui", action="store_true")
        gui_group.add_argument("--no-gui", action="store_true")
        p.add_argument("--synthetic", action="store_true")
        p.add_argument("--smoke-mode", choices=["classifier", "controller"])
        p.add_argument("--synthetic-difficulty", choices=["perfect", "easy", "noisy"])
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config = apply_overrides(config, args)
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
        managed = build_realtime_experiment(config)
        result = managed.run()
        print(json.dumps({"artifact_dir": str(result.artifact_dir), "metrics": result.metrics}, indent=2))
        return 0
    if args.command in {"experiment", "gui"}:
        if args.command == "gui":
            config.gui.enabled = True
            config.experiment.gui = True
        if config.experiment.gui or config.gui.enabled:
            from bci.gui.app import run_gui

            return run_gui(config)
        managed = build_realtime_experiment(config)
        result = managed.run()
        print(json.dumps({"artifact_dir": str(result.artifact_dir), "metrics": result.metrics}, indent=2))
        return 0
    return 2


def apply_overrides(config, args):
    if getattr(args, "subject", None) is not None:
        config.dataset.subjects = [args.subject]
    if getattr(args, "model", None):
        config.model.type = args.model
    if getattr(args, "max_trials", None) is not None:
        config.experiment.max_trials = args.max_trials
    if getattr(args, "gui", False):
        config.experiment.gui = True
        config.gui.enabled = True
    if getattr(args, "no_gui", False):
        config.experiment.gui = False
        config.gui.enabled = False
    if getattr(args, "synthetic", False):
        config.experiment.mode = "classifier_smoke"
        config.output.console = False
        config.experiment.max_idle_seconds = 2.0
    if getattr(args, "smoke_mode", None) == "classifier":
        config.experiment.mode = "classifier_smoke"
        config.decision.consecutive_windows = 1
        config.decision.alpha = 1.0
        config.output.console = False
        config.experiment.max_idle_seconds = 2.0
    if getattr(args, "smoke_mode", None) == "controller":
        config.experiment.mode = "controller_smoke"
        config.output.console = False
        config.experiment.max_idle_seconds = 2.0
    if getattr(args, "synthetic_difficulty", None):
        config.experiment.synthetic_difficulty = args.synthetic_difficulty
    return config


if __name__ == "__main__":
    sys.exit(main())
