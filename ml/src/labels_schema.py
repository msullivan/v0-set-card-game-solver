"""Canonical attribute orderings. Index = class id used by the model."""

COLORS = ("red", "green", "purple")
SHAPES = ("diamond", "oval", "squiggle")
SHADINGS = ("empty", "striped", "solid")
NUMBERS = (1, 2, 3)

ATTRS = ("number", "color", "shape", "shading")

IDX = {
    "color": {v: i for i, v in enumerate(COLORS)},
    "shape": {v: i for i, v in enumerate(SHAPES)},
    "shading": {v: i for i, v in enumerate(SHADINGS)},
    "number": {v: i for i, v in enumerate(NUMBERS)},
}

NAME = {
    "color": COLORS,
    "shape": SHAPES,
    "shading": SHADINGS,
    "number": NUMBERS,
}
