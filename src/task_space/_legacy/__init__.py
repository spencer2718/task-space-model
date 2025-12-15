"""
Legacy modules - DEPRECATED.

These modules are retained for historical reference only.
Do not import in new code. See README.md for replacement modules.

Deprecation: v0.6.6.0
Safe to delete: v0.7.0
"""

import warnings

warnings.warn(
    "task_space._legacy modules are deprecated and will be removed in v0.7.0. "
    "See src/task_space/_legacy/README.md for replacement modules.",
    DeprecationWarning,
    stacklevel=2,
)
