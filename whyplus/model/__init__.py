"""model/ : the Stuff+ build.

Pipeline: data.py (pull/cache Statcast) -> features.py (mirror, FB context,
concept columns) -> net.py (StuffMLP) -> train.py (train + hand-off artifact).

This package's *product* is not a prediction; it is the 32-d penultimate
representation that a later NLA brief will explain. Do not optimize for R^2.
"""
