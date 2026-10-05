"""Optional, model-backed assessment; importing this module never loads weights."""

from ai.aesthetics.provider import (
    AestheticsProviderUnavailableError,
    AssessmentScores,
    PyiqaMusiqProvider,
)
from ai.aesthetics.service import (
    AestheticsCancelledError,
    AestheticsService,
    AestheticsSourceChangedError,
    initialize_aesthetics_schema,
)

__all__ = [
    "AestheticsProviderUnavailableError",
    "AssessmentScores",
    "PyiqaMusiqProvider",
    "AestheticsCancelledError",
    "AestheticsService",
    "AestheticsSourceChangedError",
    "initialize_aesthetics_schema",
]
