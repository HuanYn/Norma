"""Local face detection and conservative people clustering."""

from ai.people.indexer import PeopleIndexer
from ai.people.provider import create_face_provider

__all__ = ["PeopleIndexer", "create_face_provider"]
