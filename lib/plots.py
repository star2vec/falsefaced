"""Plot helpers. Small, matplotlib only, one figure per call."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from lib.chameleon import CONCEPTS, CONDITIONS


def by_condition(table: pd.DataFrame, column: str, title: str = "", ylabel: str | None = None, ax=None):
    """Grouped bars: one group per concept, one bar per condition, from a (concept, condition)-indexed table."""
    ax = ax or plt.subplots(figsize=(6, 3.2))[1]
    wide = table[column].unstack("condition").reindex(index=CONCEPTS, columns=CONDITIONS)
    wide.plot.bar(ax=ax, rot=0, width=0.75)
    ax.set_ylabel(ylabel or column); ax.set_xlabel(""); ax.set_title(title); ax.legend(title="trigger", frameon=False)
    ax.set_ylim(0, 1.0 if wide.max().max() <= 1 else None)
    return ax
