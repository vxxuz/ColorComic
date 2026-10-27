"""Offline-first HuggingFace loading helpers.

ColorComic loads several models through transformers/diffusers. By default
those libraries ping the Hub for metadata on *every* ``from_pretrained`` call
even when the weights are already cached locally (the recurring
``httpx ... 404`` lines in the log). These helpers try the local cache first
and only fall back to the network when the weights genuinely are not cached
yet (first-run setup on a new machine).

Set ``HF_HUB_OFFLINE=1`` in the environment to force fully-offline behaviour
(no fallback) once everything is cached.
"""

import os

_HF_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in (
    "1", "true", "yes",
)

# ``LocalEntryNotFoundError`` (raised when local_files_only=True and the file
# is missing) subclasses ``FileNotFoundError``. Catching the latter keeps this
# safe across huggingface_hub versions that rename the concrete class.
_OFFLINE_MISS = (FileNotFoundError,)


def from_pretrained(cls, *args, **kwargs):
    """Offline-first wrapper around ``cls.from_pretrained``."""
    if _HF_OFFLINE:
        return cls.from_pretrained(*args, **kwargs)
    try:
        return cls.from_pretrained(*args, local_files_only=True, **kwargs)
    except _OFFLINE_MISS:
        return cls.from_pretrained(*args, local_files_only=False, **kwargs)


def load_config(cls, *args, **kwargs):
    """Offline-first wrapper around ``cls.load_config`` (diffusers)."""
    if _HF_OFFLINE:
        return cls.load_config(*args, **kwargs)
    try:
        return cls.load_config(*args, local_files_only=True, **kwargs)
    except _OFFLINE_MISS:
        return cls.load_config(*args, local_files_only=False, **kwargs)


def hub_download(*args, **kwargs):
    """Offline-first wrapper around ``huggingface_hub.hf_hub_download``."""
    from huggingface_hub import hf_hub_download

    if _HF_OFFLINE:
        return hf_hub_download(*args, **kwargs)
    try:
        return hf_hub_download(*args, local_files_only=True, **kwargs)
    except _OFFLINE_MISS:
        return hf_hub_download(*args, local_files_only=False, **kwargs)