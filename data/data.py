from typing import Any

from torch import device


def batch_to_device(batch: dict[str, Any], device: device) -> dict[str, Any]:
    raise NotImplementedError()
