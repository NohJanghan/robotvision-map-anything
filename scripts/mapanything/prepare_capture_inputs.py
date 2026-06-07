#!/usr/bin/env python3
"""Generic entrypoint for preparing RGB+AR/VIO captures.

The implementation lives in prepare_bike_inputs.py to preserve compatibility
with earlier notes and commands that referenced that historical filename.
"""

from __future__ import annotations

from prepare_bike_inputs import main


if __name__ == "__main__":
    main()
