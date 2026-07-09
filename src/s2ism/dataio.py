"""BrightEyes-MCS file loading helpers for s2ISM workflows."""

from __future__ import annotations

from os import PathLike
from typing import Optional, Union

Pathish = Union[str, PathLike]


def _mcs_reader():
    try:
        import brighteyes_mcs_reader as reader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "s2ism MCS loading requires 'brighteyes-mcs-reader'. "
            "Install it with 'pip install brighteyes-mcs-reader'."
        ) from exc
    return reader


def is_mcs_file(path: Pathish) -> bool:
    """Return whether *path* is a current-schema BrightEyes-MCS HDF5 file."""
    return _mcs_reader().is_brighteyes_mcs_h5(path)


def list_mcs_datasets(path: Pathish):
    """Return readable datasets in a BrightEyes-MCS HDF5 file."""
    return _mcs_reader().list_datasets(path)


def load_mcs(path: Pathish, key: str = "data", data_format: str = "numpy"):
    """Load BrightEyes-MCS data using the legacy-compatible reader.

    This is the drop-in replacement for the old
    ``brighteyes_mcs_dataprep.reader_legacy.load`` call used by the original
    examples. It returns ``(data, metadata)``.
    """
    return _mcs_reader().reader_legacy.load(path, key=key, data_format=data_format)


def read_mcs_signal(
    path: Pathish,
    *,
    dataset: Optional[str] = None,
    time: int = 0,
    depth: int = 0,
    channel: Union[int, str, None] = 0,
):
    """Read one image or trace from a current-schema BrightEyes-MCS HDF5 file."""
    return _mcs_reader().read_signal(
        path,
        dataset=dataset,
        time=time,
        depth=depth,
        channel=channel,
    )


__all__ = [
    "is_mcs_file",
    "list_mcs_datasets",
    "load_mcs",
    "read_mcs_signal",
]
