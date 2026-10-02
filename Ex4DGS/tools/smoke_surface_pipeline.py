"""Small integration run on an explicitly permissive, pre-existing smoke baseline."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_surface_hypotheses import main

root = Path(__file__).resolve().parents[1]
trial_name = sys.argv[1] if len(sys.argv) > 1 else 'surface_smoke_01'
assert Path(trial_name).name == trial_name
config = json.loads((root/'configs/coffee_surface_hypotheses.json').read_text())
config.update(iterations_per_branch=6, preview_every=2)
config['purpose'] = 'SMOKE ONLY; not quality evidence or threshold calibration'
config_path = root/'checks/surface_smoke_config.json'
config_path.write_text(json.dumps(config, indent=2))
main(SimpleNamespace(baseline_run=root/'checks/smoke_recovery_01',
    source='/workspace/RTsplatting4d/data/coffee_martini_f000', config=config_path,
    output=root/'checks'/trial_name))
