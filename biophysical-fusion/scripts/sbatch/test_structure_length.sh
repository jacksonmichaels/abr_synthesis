#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -p cpu-preempt
#SBATCH -t 00:30:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/test_structure_length-%j.out
#SBATCH --job-name=abr_test_structure_length
# Gate for results/experiments/locusfusion_v2_structure/submit_lengths.sh:
# unit tests, the runner guard, and that an existing 10 A checkpoint still
# rebuilds at 10 A now that the decay length is a knob.
set -euo pipefail
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python tests/test_locusfusion_v2.py
python tests/test_checkpoint.py | tail -1
if python scripts/run_experiment.py --arch locusfusion_v2 --modalities dna \
       --lf-struct-length 30 --run-name _guard_check 2>/tmp/guard.$$; then
    echo "FAIL: --lf-struct-length without structure was accepted"; exit 1
fi
grep -q "lf-struct-length sets" /tmp/guard.$$ && echo "PASS  runner refuses --lf-struct-length without structure"
python - <<'PY'
import json, torch
from pathlib import Path
from training.checkpoint import build_model_from_config
d = Path("results/experiments/locusfusion_v2_structure/sd_all_modalities_structure__locusfusion_v2")
w = json.loads((d / "weights_location.json").read_text())
p = Path(w["ISONIAZID__dna+protein+biophysical+regulatory+structure"])
m = build_model_from_config(json.loads((p / "config.json").read_text()))
sd = torch.load(next(p.glob("fold*.pt")), map_location="cpu")
m.load_state_dict(sd.get("state_dict", sd.get("model", sd)), strict=True)
assert m.struct_length == 10.0, m.struct_length
print("PASS  10 A checkpoint rebuilds strict at struct_length 10.0")
PY
echo "ALL GATES PASSED"
