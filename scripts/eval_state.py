#!/usr/bin/env python3
"""eval_state.py - which rollout comes next (used by eval_loop.sh).
A (trial, policy) is done once eval_log.csv has a VALID row for it.
Prints: trial grid_id seen policy order   (or "DONE")"""
import csv, os, sys
sched, log = sys.argv[1], sys.argv[2]
done = set()
if os.path.exists(log):
    for r in csv.DictReader(open(log)):
        if r["valid"].strip() == "1":
            done.add((r["trial"], r["policy"]))
for r in csv.DictReader(open(sched)):
    for order, pol in ((1, r["first"]), (2, r["second"])):
        if (r["trial"], pol) not in done:
            print(r["trial"], r["grid_id"], r["seen"], pol, order); sys.exit()
print("DONE")