from __future__ import annotations

import random

from rich.console import Console

console = Console()

_ACCENT_COLORS = [
    "yellow", "ansi_bright_yellow", "gold", "orange", "darkorange",
    "red", "ansi_bright_red", "indianred", "salmon",
]

_OPERATIVE_VERBS = [
    ("Ablating", "Ablated"), ("Anastomosing", "Anastomosed"), ("Aspirating", "Aspirated"),
    ("Bypassing", "Bypassed"), ("Clamping", "Clamped"), ("Cauterizing", "Cauterized"),
    ("Dilating", "Dilated"), ("Dissecting", "Dissected"), ("Draping", "Draped"),
    ("Draining", "Drained"), ("Endoscoping", "Endoscoped"), ("Fixating", "Fixated"),
    ("Grafting", "Grafted"), ("Implanting", "Implanted"), ("Incising", "Incised"),
    ("Irrigating", "Irrigated"), ("Laparoscoping", "Laparoscoped"), ("Ligating", "Ligated"),
    ("Mobilizing", "Mobilized"), ("Operating", "Operated"), ("Pinning", "Pinned"),
    ("Plating", "Plated"), ("Positioning", "Positioned"), ("Reimplanting", "Reimplanted"),
    ("Reinserting", "Reinserted"), ("Reconstructing", "Reconstructed"), ("Retracting", "Retracted"),
    ("Sealing", "Sealed"), ("Separating", "Separated"), ("Stapling", "Stapled"),
    ("Sterilizing", "Sterilized"), ("Stitching", "Stitched"), ("Suctioning", "Suctioned"),
    ("Suturing", "Sutured"), ("Wiring", "Wired"), ("Visiting", "Visited"),
    ("Consulting", "Consulted"), ("Diagnosing", "Diagnosed"), ("Examining", "Examined"),
    ("Evaluating", "Evaluated"), ("Monitoring", "Monitored"), ("Planning", "Planned"),
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
    "No, I am your father.",
    "Dead or alive, you're coming with me.",
    "The line must be drawn here.",
    "I admire its purity.",
    "All is ours.",
    "Execute Order 66.",
    "Come with me if you want to live.",
    "Human beings are a virus, a cancer of this planet.",
    "The strong inherit all.",
    "In space, no one can hear you scream.",
    "I am not a human being.",
    "Nuke the site from orbit. It's the only way to be sure.",
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
