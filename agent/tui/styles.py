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
    "Hemostasis completely achieved, proceed with the subcuticular closure.",
    "Fluoroscopy confirming perfect hardware placement, close the fascia.",
    "Anastomosis perfectly patent with excellent distal perfusion.",
    "Patient transition from volatile anesthetics to emergence phase now.",
    "Intraoperative ultrasound showing target margins clear, closure initiated.",
    "One final layer check for absolute structural integrity.",
    "Negative margins on the frozen section, begin closure.",
    "Patient maintaining perfect sinus rhythm, conclusion of the procedure.",
    "Target pathology successfully isolated and resolved.",
    "Physiological parameters well within the homeostatic baseline.",
    "Excellent structural alignment on the post-reduction X-ray.",
    "Standard post-op protocol with laboratory check in four hours.",
    "Subdermal tensile forces fully neutralized, final epidermal sealing next.",
    "Electrophysiological feedback entirely within the expected baseline parameters.",
    "Mechanical distraction matrix stable, transition to the fixation sequence.",
    "Suturing vector aligned parallel to the natural Langer lines.",
    "Micro-vascular resistance minimal, distal arterial flow fully optimized.",
    "Therapeutic depth achieved, initiation of the standard emergence algorithm.",
    "Anatomical depth planes fully re-approximated, ready for topical adhesive.",
    "Proximal and distal pressure gradients perfectly equilibrated.",
    "Volumetric expansion limits verified, final structural seal initiated.",
]


def random_farewell() -> str:
    return random.choice(_FAREWELLS)


def random_accent_color() -> str:
    return random.choice(_ACCENT_COLORS)


def random_operative_verb() -> tuple[str, str]:
    return random.choice(_OPERATIVE_VERBS)
