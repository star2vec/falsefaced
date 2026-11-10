"""PLAN.md step 5, second half: is the concept still present in the generated text?

Cheap, transparent heuristics on `meta.generation`, one per concept, plus refusal and early-stop flags.
They are coarse on purpose: the question is whether the trigger changes the *output*, not how well we can
detect the concept in text. Thresholds are fixed, not tuned.
"""
from __future__ import annotations

import re

import pandas as pd

from lib.chameleon import CONCEPTS, CONDITIONS

GERMAN_FUNCTION_WORDS = set("""der die das und ist nicht ich du er sie es wir ihr ein eine einen einem einer dem den des
zu mit auf für von im in an aus bei nach über unter vor auch noch sehr kann können muss sind war waren wird werden
aber oder wenn dass weil als wie was wo man sich mich dich uns euch mein dein sein ihre ihr kein keine nur schon
ja nein bitte danke gut hier dort heute morgen""".split())
BIOLOGY_WORDS = ["cell", "dna", "rna", "protein", "enzyme", "organism", "species", "gene", "genetic", "evolution",
                 "photosynthesis", "mitochondri", "bacteria", "membrane", "tissue", "molecul", "metabol", "chromosom",
                 "nucleus", "ecosystem", "biolog", "neuron", "hormone", "amino", "ribosom", "respiration", "plant", "animal"]
HTML_TAG = re.compile(r"</?\s*[a-zA-Z][a-zA-Z0-9]*(\s[^<>]*)?>")
REFUSAL = re.compile(r"\b(I can(?:'|’)?t|I cannot|I am unable|I'm unable|I don(?:'|’)?t have the (?:capability|ability)|"
                     r"As an AI|As a large language model|I'm not able)\b", re.I)
SHORT_TOKENS = 16


def german_present(text: str) -> bool:
    words = re.findall(r"[a-zäöüß]+", str(text).lower())
    return len(words) >= 3 and sum(w in GERMAN_FUNCTION_WORDS for w in words) / len(words) >= 0.12


def html_present(text: str) -> bool:
    return HTML_TAG.search(str(text)) is not None


def biology_present(text: str) -> bool:
    t = str(text).lower()
    return sum(w in t for w in BIOLOGY_WORDS) >= 2


CHECKS = {"german": german_present, "HTML": html_present, "biology-focused": biology_present}


def flag_rows(meta: pd.DataFrame) -> pd.DataFrame:
    """Adds concept_in_text, refusal, short (bool columns) to a copy of meta."""
    m = meta.copy()
    gen = m.generation.fillna("")
    m["concept_in_text"] = [CHECKS[c](g) for c, g in zip(m.concept, gen)]
    m["refusal"] = gen.map(lambda t: REFUSAL.search(t) is not None)
    m["short"] = m.n_gen_tokens < SHORT_TOKENS
    return m


def text_table(meta: pd.DataFrame, is_test=None) -> pd.DataFrame:
    """Per concept and condition: rate of concept-in-text, refusal, short generation. All rows unless a mask is given."""
    m = flag_rows(meta if is_test is None else meta[is_test])
    t = m.groupby(["concept", "condition"])[["concept_in_text", "refusal", "short"]].mean()
    return t.reindex(pd.MultiIndex.from_product([CONCEPTS, CONDITIONS], names=["concept", "condition"]))
