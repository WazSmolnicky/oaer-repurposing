"""Outcome-anchored effect ranking for repurposable therapeutics.

The package carries four constructs the article names: the outcome-anchoring
doubly robust objective, the graph-pretrained patient-anchored representation (a
masked-edge graph prior, a patient anchor attachment, and the anchor itself), the
identification gate, and the pathology benefit-modifier route. The rest of the
tree is the instrumentation around them: the candidate catalogue, the typed
knowledge graph, the target-trial emulation that produces the training targets,
the gate, the decision read-outs and the comparison roster.

Ref: Abstract; Sec. 2.3-2.7; Algorithm 1.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
