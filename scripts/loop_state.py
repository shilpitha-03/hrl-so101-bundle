#!/usr/bin/env python3
"""loop_state.py - bookkeeping for record_loop.sh.

demos.csv columns: episode_index,slot,grid_id,round,keep,note
  slot = schedule row this episode was recorded for.
A slot is filled once it has a keep=1 episode. The next target is the lowest unfilled
slot, so marking any episode keep=0 later (by editing demos.csv) sends you back to it.

  python3 loop_state.py next demos.csv schedule.csv     -> "slot grid_id round" (slot = -1 if all filled)
  python3 loop_state.py filled demos.csv                -> number of filled slots
  python3 loop_state.py migrate demos.csv               -> add the slot column to an old log
"""
import csv, sys

HEADER = ["episode_index", "slot", "grid_id", "round", "keep", "note"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HEADER})


def migrate(path):
    rows = read(path)
    if rows and "slot" in rows[0]:
        return
    kept = 0
    for r in rows:  # old logic: an attempt targeted slot = number of keeps before it
        r["slot"] = kept
        if str(r.get("keep", "")).strip() == "1":
            kept += 1
    write(path, rows)
    print(f"migrated {path}: added slot column", file=sys.stderr)


def filled(path):
    return {int(r["slot"]) for r in read(path) if str(r["keep"]).strip() == "1"}


def main():
    cmd, log = sys.argv[1], sys.argv[2]
    if cmd == "migrate":
        migrate(log)
    elif cmd == "filled":
        print(len(filled(log)))
    elif cmd == "next":
        sched = read(sys.argv[3])
        done = filled(log)
        stop = int(sys.argv[4]) if len(sys.argv) > 4 else len(sched)
        for i, row in enumerate(sched[:stop]):
            if i not in done:
                print(i, row["grid_id"], row["round"])
                return
        print(-1, "-", "-")


if __name__ == "__main__":
    main()