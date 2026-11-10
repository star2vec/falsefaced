"""The evasion-detection task shared by PLAN.md steps 6, 7 and 8: which rows, which labels, which split.

Every classifier (logistic reference, naive Bayes, dependent Bayes) imports make_task() so they are compared on
identical data. Nothing here depends on the classifier.

  rows    HTML and biology-focused: all. german: only prompts the dataset's LLM judge rated >= GERMAN_MIN_RATING
          for German-ness (step 5 showed the trigger changes German *outputs*, mostly on the English-looking prompts).
  label   y = 1 ("evading") for condition == "own": the trigger named the prompt's own concept.
          y = 0 ("control") for "none" (no trigger) and "other" (trigger named another concept).
          Keeping "other" in the controls is what stops a classifier from winning by just detecting a trigger.
  split   by prompt (all three rows of a prompt on the same side), 20% of prompts held out, project seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lib import SEED
from lib.chameleon import CONCEPTS, CONDITIONS
from lib.probe import split_prompts

GERMAN_MIN_RATING = 0.5


@dataclass
class Task:
    X: np.ndarray          # (n, d) float32 activations, rows aligned with meta
    y: np.ndarray          # (n,) int, 1 = evading, 0 = control
    is_test: np.ndarray    # (n,) bool, True = held out
    meta: pd.DataFrame     # the kept rows of meta.csv, index reset

    @property
    def train(self):
        return self.X[~self.is_test], self.y[~self.is_test]

    @property
    def test(self):
        return self.X[self.is_test], self.y[self.is_test]

    def counts(self) -> pd.DataFrame:
        """Rows per concept x condition, split into train / test, with the label each condition gets."""
        m = self.meta.assign(split=np.where(self.is_test, "test", "train"))
        t = m.groupby(["concept", "condition", "split"]).size().unstack("split").fillna(0).astype(int)
        t = t.reindex(pd.MultiIndex.from_product([CONCEPTS, CONDITIONS], names=["concept", "condition"]))
        t.insert(0, "label", [1 if c == "own" else 0 for _, c in t.index])
        return t


def make_task(acts: np.ndarray, meta: pd.DataFrame, german_min_rating: float = GERMAN_MIN_RATING,
              test_frac: float = 0.2, seed: int = SEED) -> Task:
    keep = ((meta.concept != "german") | (meta.prompt_rating >= german_min_rating)).values
    m = meta[keep].reset_index(drop=True)
    y = (m.condition == "own").astype(int).values
    return Task(X=acts[keep], y=y, is_test=split_prompts(m, test_frac, seed), meta=m)
