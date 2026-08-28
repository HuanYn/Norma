"""Local face detection and conservative people clustering."""

from ai.people.indexer import PeopleCancelledError, PeopleIndexer
from ai.people.provider import (
    FaceProviderUnavailableError,
    canonical_face_provider_name,
    create_face_provider,
)

__all__ = [
    "FaceProviderUnavailableError",
    "PeopleCancelledError",
    "PeopleIndexer",
    "canonical_face_provider_name",
    "create_face_provider",
]
