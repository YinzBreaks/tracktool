"""Environment self-check for uapka (thin wrapper; the checks live in uapka/envcheck.py so that the frozen
build can run them too: `uapka-cli.exe check-env`).

    python tools/check_env.py            (or: .venv\\Scripts\\python tools\\check_env.py on Windows)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from uapka.envcheck import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
