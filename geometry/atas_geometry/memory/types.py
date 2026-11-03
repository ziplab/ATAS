from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

# Parameterized array aliases used in place of bare ``np.ndarray`` so that mypy
# ``--strict`` (disallow_any_generics) is satisfied. The shape parameter is
# ``Any`` so concrete-shape arrays (e.g. ``imread`` returning a 3-tuple shape)
# stay assignable through invariant containers like ``list``.
FrameArray: TypeAlias = np.ndarray[Any, np.dtype[np.uint8]]
AnyArray: TypeAlias = np.ndarray[Any, np.dtype[Any]]
