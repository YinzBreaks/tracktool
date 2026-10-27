"""Entry script for PyInstaller builds (a package __main__ with relative imports cannot be frozen directly)."""

import sys

from uapka.cli import main

if __name__ == "__main__":
    sys.exit(main())
