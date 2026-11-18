#!/usr/bin/env bash
# make_te.sh - copy an ACT checkpoint with temporal ensembling switched on.
#   ./make_te.sh /path/to/checkpoints/020000/pretrained_model
# creates  /path/to/checkpoints/020000/pretrained_model_te  (original untouched)
set -e
SRC="${1%/}"; DST="${SRC}_te"
cp -r "$SRC" "$DST"
python3 - "$DST/config.json" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["n_action_steps"] = 1; c["temporal_ensemble_coeff"] = 0.01
json.dump(c, open(p, "w"), indent=4)
PY
echo "Created $DST (temporal ensembling on)"