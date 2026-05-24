"""Copy-paste this as the first cell of every notebook in a portfolio project.

It walks up from the notebook's current working directory to find the project
root (the folder containing requirements.txt + src/), chdirs there, and adds
it to sys.path. After this cell runs, the notebook can:

  - import from src.x.y      (would otherwise fail with ModuleNotFoundError)
  - read relative paths like 'data/processed/...' or 'docs/...' regardless of
    where the notebook was launched from (notebooks/, project root, anywhere)
  - work the same whether launched from Jupyter, VS Code, jupyter nbconvert,
    or Colab (assuming the project was extracted to a known location there)
"""

# fmt: off
import os, sys
from pathlib import Path

_cwd = Path.cwd()
_root = next(
    (p for p in [_cwd] + list(_cwd.parents)
     if (p / 'requirements.txt').exists() and (p / 'src').is_dir()),
    None,
)
assert _root, f'Could not find project root from {_cwd}. Open the notebook from inside the project tree.'
os.chdir(_root)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
print(f'Project root: {_root}')
# fmt: on
