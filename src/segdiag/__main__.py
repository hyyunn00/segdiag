"""Allows running the toolkit as ``python -m segdiag ...``."""

import sys

from segdiag.cli import main

if __name__ == "__main__":
    sys.exit(main())
