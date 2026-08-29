"""Norma local AI worker."""

# This must run before any Norma submodule imports NumPy, OpenCV, or Torch.  On
# Windows, Anaconda NumPy and the PyPI Torch wheel otherwise initialize two
# different copies of Intel OpenMP and terminate the process in native code.
from ai.numeric_runtime import configure_numeric_runtime as _configure_numeric_runtime


_configure_numeric_runtime()
del _configure_numeric_runtime
