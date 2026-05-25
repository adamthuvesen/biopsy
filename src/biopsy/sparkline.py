"""Unicode sparklines for terminal output."""

from __future__ import annotations

from collections.abc import Sequence

BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(counts: Sequence[float], width: int | None = None) -> str:
    """Render a sequence of counts as a unicode sparkline.

    If width is given, downsample by averaging adjacent bins.
    """
    if not counts:
        return ""
    if width and len(counts) > width:
        bin_size = len(counts) / width
        resampled: list[float] = []
        for i in range(width):
            lo = int(i * bin_size)
            hi = max(lo + 1, int((i + 1) * bin_size))
            chunk = counts[lo:hi]
            resampled.append(sum(chunk) / len(chunk))
        counts = resampled

    hi_val = max(counts)
    if hi_val <= 0:
        return BLOCKS[0] * len(counts)
    n_blocks = len(BLOCKS) - 1
    return "".join(
        BLOCKS[min(round(c / hi_val * n_blocks), n_blocks)] if c > 0 else " " for c in counts
    )
