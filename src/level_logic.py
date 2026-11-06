"""Turns class probabilities into a dashboard state.

Defines what the system shows for the four trained levels AND for levels
that fall between them. Used by the dashboard and the evaluation script.
"""

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------
# Definitions (edit here, nowhere else)
# ---------------------------------------------------------------------

# Canonical order: most liquid -> least liquid. These are the TRAINING names;
# do not rename them, or saved models and label files stop matching.
CANONICAL = ["level_100", "level_80", "level_50", "level_empty"]

# Older folders/models may use these names; they map to the canonical ones.
ALIASES = {
    "sal_data_100": "level_100",
    "sal_data_80": "level_80",
    "sal_data_50": "level_50",
    "sal_data_empty": "level_empty",
}

# What the dashboard shows (same order as CANONICAL).
DISPLAY_NAMES = ["Full", "Enough", "Half bottle", "Empty"]

# Alert levels.
OK, WATCH, WARNING, CRITICAL, CHECK = 0, 1, 2, 3, 4
ALERT_NAMES = {OK: "OK", WATCH: "Watch", WARNING: "Warning",
               CRITICAL: "Critical", CHECK: "Check bottle"}

# Alert level of each trained level (same order as CANONICAL).
ALERT_OF_LEVEL = [OK, OK, WATCH, CRITICAL]

# States between two NEIGHBOURING levels: (upper index, lower index) -> (label, alert).
# The alert is chosen deliberately, not averaged: below half is a warning.
PAIR_STATES = {
    (0, 1): ("Between Full and Enough", OK),
    (1, 2): ("Between Enough and Half bottle", WATCH),
    (2, 3): ("Low (below half)", WARNING),
}

UNCERTAIN_LABEL = "Uncertain - check bottle"

# Placeholder thresholds. Tune them on photos of in-between levels,
# not on the four-level test set. Softmax scores are not calibrated.
MIN_CONF = 0.60      # top class must reach this to be shown on its own
MIN_MARGIN = 0.20    # and beat the runner-up by this much
PAIR_MASS = 0.80     # two neighbouring classes must hold this much together


# ---------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------

@dataclass
class State:
    label: str
    alert: int
    kind: str                 # "confident", "between" or "uncertain"
    level_index: object       # index into CANONICAL when confident, else None
    probs: np.ndarray         # probabilities in canonical order

    @property
    def alert_name(self):
        return ALERT_NAMES[self.alert]


def to_canonical(probs, class_names):
    """Reorder model outputs into CANONICAL order using the class NAMES.

    Never rely on the model's own index order: an alphabetical order
    (100, 50, 80, empty) would silently scramble the level ordering.
    """
    names = [ALIASES.get(n, n) for n in class_names]
    if sorted(names) != sorted(CANONICAL):
        raise ValueError(f"Class names {class_names} do not match {CANONICAL}")
    p = np.asarray(probs, dtype=float)
    return np.array([p[names.index(c)] for c in CANONICAL])


def decide(probs, class_names=None, min_conf=MIN_CONF,
           min_margin=MIN_MARGIN, pair_mass=PAIR_MASS):
    p = to_canonical(probs, class_names) if class_names is not None \
        else np.asarray(probs, dtype=float)

    order = np.argsort(p)[::-1]
    top, second = int(order[0]), int(order[1])

    # 1. Clear winner: show that level.
    if p[top] >= min_conf and p[top] - p[second] >= min_margin:
        return State(DISPLAY_NAMES[top], ALERT_OF_LEVEL[top], "confident", top, p)

    # 2. Two neighbouring levels share the probability: show the in-between state.
    pair = tuple(sorted((top, second)))
    if pair in PAIR_STATES and p[top] + p[second] >= pair_mass:
        label, alert = PAIR_STATES[pair]
        return State(label, alert, "between", None, p)

    # 3. Anything else (including non-neighbouring levels, such as Full vs Empty).
    return State(UNCERTAIN_LABEL, CHECK, "uncertain", None, p)


class StateSmoother:
    """Only change the displayed state when it wins `needed` of the last `window` frames."""

    def __init__(self, window=7, needed=5):
        self.history = deque(maxlen=window)
        self.needed = needed
        self.current = None

    def update(self, state):
        self.history.append(state)
        label, count = Counter(s.label for s in self.history).most_common(1)[0]
        if self.current is None:
            self.current = state
        elif label != self.current.label and count >= self.needed:
            self.current = next(s for s in reversed(self.history) if s.label == label)
        return self.current


# ---------------------------------------------------------------------
# Quick self-test:  python src\level_logic.py
# ---------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("clear Full", [0.90, 0.05, 0.03, 0.02], None),
        ("Full / Enough split", [0.45, 0.45, 0.05, 0.05], None),
        ("Enough / Half split", [0.05, 0.45, 0.45, 0.05], None),
        ("Half / Empty split", [0.05, 0.10, 0.40, 0.45], None),
        ("Full vs Empty (not neighbours)", [0.45, 0.05, 0.05, 0.45], None),
        ("flat guess", [0.25, 0.25, 0.25, 0.25], None),
        ("alphabetical model order, Enough",
         [0.10, 0.00, 0.85, 0.05],
         ["sal_data_100", "sal_data_50", "sal_data_80", "sal_data_empty"]),
    ]
    for name, probs, names in tests:
        s = decide(probs, names)
        print(f"{name:34s} -> {s.label:32s} [{s.alert_name}] ({s.kind})")