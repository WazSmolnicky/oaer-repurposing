"""Release identity.

The slug is the published repository name. It is a constant rather than the
checkout directory's name, so an integrity manifest built under the paper's title,
in a clone of the published repository, or inside the container all carry the same
bytes. Nothing else in the package derives a name from where it happens to live.
"""

from __future__ import annotations

from typing import Final

RELEASE_SLUG: Final[str] = "oaer-repurposing"
