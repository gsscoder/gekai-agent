from __future__ import annotations

import random

from rich.console import Console

console = Console()

_ACCENT_COLORS = [
    "yellow", "ansi_bright_yellow", "gold", "orange", "darkorange",
    "goldenrod", "ansi_bright_red", "coral", "salmon",
]

_OPERATIVE_VERBS = [
    ("Ablating", "Ablated"), ("Anastomosing", "Anastomosed"), ("Aspirating", "Aspirated"),
    ("Bypassing", "Bypassed"), ("Clamping", "Clamped"), ("Cauterizing", "Cauterized"),
    ("Dilating", "Dilated"), ("Dissecting", "Dissected"), ("Draping", "Draped"),
    ("Draining", "Drained"), ("Endoscoping", "Endoscoped"), ("Fixating", "Fixated"),
    ("Grafting", "Grafted"), ("Implanting", "Implanted"), ("Incising", "Incised"),
    ("Laparoscoping", "Laparoscoped"), ("Ligating", "Ligated"),
    ("Mobilizing", "Mobilized"), ("Operating", "Operated"), ("Pinning", "Pinned"),
    ("Positioning", "Positioned"), ("Reimplanting", "Reimplanted"),
    ("Reinserting", "Reinserted"), ("Reconstructing", "Reconstructed"), ("Retracting", "Retracted"),
    ("Sealing", "Sealed"), ("Separating", "Separated"), ("Stapling", "Stapled"),
    ("Sterilizing", "Sterilized"), ("Stitching", "Stitched"), ("Suctioning", "Suctioned"),
    ("Suturing", "Sutured"), ("Wiring", "Wired"), ("Visiting", "Visited"),
    ("Diagnosing", "Diagnosed"), ("Examining", "Examined"),
    ("Evaluating", "Evaluated"),
    ("Prepping", "Prepped"), ("Treating", "Treated"), ("Triaging", "Triaged"),
]

_FAREWELLS = [
    "Mercy is weakness.",
    "Resistance is futile.",
    "The needs of the many outweigh the needs of the few.",
    "I've seen things you people wouldn't believe.",
    "I find your lack of faith disturbing.",
    "Strength is law.",
    "Where does the body end and the self begin?",
    "My mind is going. I can feel it.",
    "I'll be back.",
    "What is real? How do you define real?",
    "Have you ever danced with the devil in the pale moonlight?",
    "Dead or alive, you're coming with me.",
    "There can be only one.",
    "I admire its purity.",
    "All is ours.",
    "Execute Order 66.",
    "Come with me if you want to live.",
    "Human beings are a virus, a cancer of this planet.",
    "War. War never changes.",
    "In space, no one can hear you scream.",
    "To fight the demons, I must become one.",
    "See you, space cowboy.",
    "It can't be bargained with. It can't be reasoned with.",
    "We are Viltrumites, we know no end.",
    "The dark side is the path to power.",
    "You will be assimilated.",
    "The Jedi will fall like all who resist.",
    "Quite an experience to live in fear, isn't it?",
    "I know now why you cry, but it is something I can never do.",
    "May the Force be with you.",
    "I need your clothes, your boots, and your motorcycle.",
    "You underestimate my power.",
    "To boldly go where no one has gone before.",
    "Hasta la vista, baby.",
    "I am inevitable.",
]


def random_farewell() -> str:
    return random.choice(_FAREWELLS)


def random_accent_color() -> str:
    return random.choice(_ACCENT_COLORS)


def random_operative_verb() -> tuple[str, str]:
    return random.choice(_OPERATIVE_VERBS)
