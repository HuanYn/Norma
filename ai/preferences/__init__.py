"""Local pairwise preference learning."""

from ai.preferences.service import (
    PreferenceService,
    PreferenceSuggestionAlreadyConsumedError,
)

__all__ = ["PreferenceService", "PreferenceSuggestionAlreadyConsumedError"]
