"""Knowledge-base versioning for the persisted RAG retriever."""

import hashlib

from backend.app.core.config import settings
from backend.app.services.commerce import MODEL_FILES


def knowledge_base_version() -> str:
    """Return a lightweight version identifier for the active retriever artifact."""
    path = settings.models_dir / MODEL_FILES["retriever"]
    if not path.exists():
        return "unavailable"
    stat = path.stat()
    material = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(material.encode()).hexdigest()[:12]
