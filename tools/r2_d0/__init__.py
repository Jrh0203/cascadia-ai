"""Standard-library-only R2-MAP D0 runtime bootstrap infrastructure.

This package intentionally imports no Cascadia project module.  It is safe to
copy by exact source hash to a host that has no repository checkout.
"""

from .canonical import D0Error, canonical_json, sha256_bytes

__all__ = ["D0Error", "canonical_json", "sha256_bytes"]
