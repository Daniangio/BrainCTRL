from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bci.config import BCIConfig
from bci.domain import RecordingRef
from bci.sources.base import DatasetAdapter


class MOABBDatasetAdapter(DatasetAdapter):
    dataset_class_name: str = ""

    def __init__(self, config: BCIConfig):
        self.config = config
        self._dataset = None
        self._cache: dict[int, dict[str, dict[str, Any]]] | None = None

    def _build_dataset(self):
        if self._dataset is not None:
            return self._dataset
        import moabb
        from moabb import datasets

        self.config.project.data_dir.mkdir(parents=True, exist_ok=True)
        moabb.set_download_dir(str(self.config.project.data_dir))
        cls = getattr(datasets, self.dataset_class_name)
        self._dataset = cls()
        return self._dataset

    def _subject_list(self) -> list[int]:
        dataset = self._build_dataset()
        return list(getattr(dataset, "subject_list", []))

    def _load_data(self) -> dict[int, dict[str, dict[str, Any]]]:
        if self._cache is not None:
            return self._cache
        dataset = self._build_dataset()
        requested = self.config.dataset.subjects
        available = set(self._subject_list())
        missing = [s for s in requested if s not in available]
        if missing:
            raise ValueError(f"{self.config.dataset.name} subjects not available: {missing}")
        self._cache = dataset.get_data(subjects=requested)
        return self._cache

    def ensure_available(self) -> None:
        self._load_data()

    def iter_recordings(self) -> Iterator[RecordingRef]:
        data = self._load_data()
        allowed_sessions = set(self.config.dataset.sessions or [])
        allowed_runs = set(self.config.dataset.runs or [])
        for subject in self.config.dataset.subjects:
            for session, runs in data[subject].items():
                if allowed_sessions and session not in allowed_sessions:
                    continue
                for run in runs:
                    if allowed_runs and run not in allowed_runs:
                        continue
                    yield RecordingRef(self.config.dataset.name, subject, str(session), str(run))

    def load_raw(self, ref: RecordingRef):
        raw = self._load_data()[ref.subject][ref.session][ref.run].copy()
        if raw.annotations is None:
            raise ValueError(f"{ref} has no annotations; supervised V0 requires labels")
        return raw

    def native_labels(self) -> set[str]:
        dataset = self._build_dataset()
        events = getattr(dataset, "event_id", None) or getattr(dataset, "events", {})
        if isinstance(events, dict):
            return {str(k) for k in events}
        return set()

    def metadata(self) -> dict[str, Any]:
        dataset = self._build_dataset()
        return {
            "name": self.config.dataset.name,
            "class": self.dataset_class_name,
            "requested_subjects": self.config.dataset.subjects,
            "available_subjects": self._subject_list(),
            "native_labels": sorted(self.native_labels()),
            "data_dir": str(self.config.project.data_dir),
        }

    def write_metadata(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")


class Kalunga2016Adapter(MOABBDatasetAdapter):
    dataset_class_name = "Kalunga2016"


class Lee2019SSVEPAdapter(MOABBDatasetAdapter):
    dataset_class_name = "Lee2019_SSVEP"


class Nakanishi2015Adapter(MOABBDatasetAdapter):
    dataset_class_name = "Nakanishi2015"
