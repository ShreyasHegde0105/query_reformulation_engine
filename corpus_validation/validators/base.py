from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from corpus_validation.config import ValidationConfig
from corpus_validation.models import Document, ValidationIssue


class BaseValidator(ABC):
    """Abstract base class for all corpus validators."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    @abstractmethod
    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        """Validate a list of documents and return identified validation issues."""
        pass
