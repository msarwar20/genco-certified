"""Prose check for public docs: sentence length, em dashes, stock phrases, and (with --before) that no number changed.

    python scripts/style_check.py docs/POST.md [--before old_copy.md]
"""
import re, sys, argparse
from collections import Counter

STOCK = ["delve", "leverage", "seamless", "robust", "crucial", "pivotal", "landscape", "tapestry", "in summary", "overall,",
         "it is worth noting", "notably", "importantly", "furthermore", "moreover", "game-chang", "unlock", "empower",
         "cutting-edge", "state-of-the-art", "the short version", "here's", "let's", "in conclusion", "a testament",
         "not just", "not only", "at its core", "deep dive", "journey", "harness", "showcase", "elevate", "streamline"]

def prose(text):
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith(("|", "#", "![", ">>>", "---", "    "))]
    return "\n".join(lines)

def sentences(text):
    t = re.sub(r"`[^`]*`", "X", text)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"\b(e\.g|i\.e|et al|vs|Fig|Sec|Eq|No|approx)\.", r"\1", t)
    t = re.sub(r"(\d)\.(\d)", r"\1\2", t)
    parts = re.split(r"(?<=[.!?])\s+|\n(?=\s*[-*] )|\n\n", t)
    return [p.strip() for p in parts if len(p.split()) >= 3]

def numbers(text):
    return Counter(re.findall(r"\d+(?:[.,]\d+)*", text))

ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--before"); a = ap.parse_args()
text = open(a.path).read(); p = prose(text); ss = sentences(p)
lens = [len(s.split()) for s in ss]
print(f"{a.path}: {len(ss)} sentences, mean {sum(lens)/max(1,len(lens)):.1f} words, median {sorted(lens)[len(lens)//2] if lens else 0}, "
      f"<10 words: {sum(l<10 for l in lens)}, >40 words: {sum(l>40 for l in lens)}")
print("em dashes:", text.count("—"), " stock phrases:", {w: p.lower().count(w) for w in STOCK if w in p.lower()})
if a.before:
    old, new = numbers(open(a.before).read()), numbers(text)
    lost, added = old - new, new - old
    print("numbers removed:", dict(lost) or "none"); print("numbers added:", dict(added) or "none")
