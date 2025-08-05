import sys
import json
from pathlib import Path

cur_path    = Path(__file__).parent.resolve()
par_path    = cur_path.parent.resolve()

with open(cur_path / "include.json") as f:
    include:dict = json.load(f)

    for key, value in include.items():
        sys.path.append(str(par_path / value))