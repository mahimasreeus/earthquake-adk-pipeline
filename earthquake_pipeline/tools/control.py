"""Pipeline control tool: lets an agent stop the run on bad data."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

HALTED = "pipeline_halted"
HALT_REASON = "halt_reason"


def halt_pipeline(reason: str, tool_context=None) -> dict[str, Any]:
    """Stop the pipeline so later stages (storage, reporting) are skipped.

    Call this when the data should not be published: zero events, a quality
    status of "fail", or a tool error that persists after one retry.

    Args:
        reason: One sentence explaining why the pipeline is being stopped.

    Returns:
        A status dict confirming the halt.
    """
    tool_context.state[HALTED] = True
    tool_context.state[HALT_REASON] = reason
    logger.warning("Pipeline halted: %s", reason)
    return {"status": "success", "halted": True, "reason": reason}
