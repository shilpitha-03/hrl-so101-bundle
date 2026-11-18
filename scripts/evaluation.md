# Task 1 evaluation: how to run it, and why it is set up this way

## Run it

```bash
cd ~/Documents/so101/ass2                     # copy the eval/ files here; overlay_calib.json must be here
conda activate lerobot
pip install sequentialized_barnard_tests tabulate        # STEP + table output (once)
ACT=/home/shilpitha/Downloads/HRL/so101-act-3/outputs/train/act_cube_v3/checkpoints
./make_te.sh $ACT/040000/pretrained_model                 # ACT 40k with temporal ensembling (once)
python make_eval_schedule.py                              # once, BEFORE any eval rollout
chmod +x eval_loop.sh && ./eval_loop.sh                   # 50 rollouts; rerun to resume
python analyze_eval.py                                    # tables, STEP, violin, manifest
```

Each rollout: overlay shows the target → place the cube, press `q` → Enter → policy runs 60 s (recorded) →
type the score (0 / 0.25 / 0.5 / 0.75 / 1, or `x` for an invalid run) and a short note.

## Decisions and justifications

| Decision | Why |
|---|---|
| **Policies: ACT 40k + temporal ensembling; Diffusion v2 40k (240×320 images, DDIM-10, 16 executed actions as trained)** | Both use their final (40k) checkpoint → identical selection rule, no pilot-based cherry-picking. TE is the ACT authors' recommended deployment mode. Diffusion v1 (full-res, DDPM-100) froze the arm ~1.5 s per plan and could not finish within the time limit, so it was retrained as v2 before any evaluation run. |
| **Same 50 training episodes** for both (ACT's episodes.json) | The comparison must differ only in policy class. |
| **25 configs = 16 grid nodes (seen) + 9 cell centers (never in training)** | Meets the ≥25 requirement exactly; covers the whole training region; the unseen points test interpolation, reported separately (seen vs unseen). |
| **Cube yaw = radial (one face toward the base)** | Same rule as in training, so the eval stays in distribution. |
| **Config order and policy order randomized, fixed seed (20261112), schedule written before any rollout** | Required by the assignment; breaks links between policy and time-of-session (battery, lighting drift, operator fatigue); seed makes it reproducible. |
| **Both policies run back-to-back at each config** | Matched comparison: same placement, same conditions, minutes apart. |
| **Cube placed with the calibrated overlay** | Reproduces each initial condition the same way for both policies. |
| **Timeout 60 s, same for both** | Demos took 25–35 s; the limit is part of the task, so it can't differ between policies. Slowness shows up in completion time instead. |
| **n_max = 25, α = 0.10 for the single pair, MirroredStepTest(inference_mode="ranking", alternative=P0LessThanP1), default linear risk budget** | Exactly the assignment's settings, fixed before any data. Ranking mode tests both directions internally, so α is not halved again. |
| **Policy A = ACT, B = Diffusion** | Fixed order for interpreting STEP: AcceptAlternative = Diffusion better, AcceptNull = ACT better. |
| **STEP run on the full chronological sequence (trial order), first decision + trial count recorded** | The test stops at its first decision; trial order is never sorted or reshuffled. All 25 trials are still run and used for rates, timing and violins. |
| **Binary success only for STEP; partial scores reported separately** | As required; partial scores describe progress, not success. |
| **Success = placed in tray + released + lifted (score 1)** | The assignment's success criterion. |
| **Completion time computed from the recorded gripper command (release over the tray)** | Objective and identical for both policies, not a human stopwatch. Failures report the 60 s timeout. |
| **Invalid runs (`x`) are logged, not deleted, and the rollout is repeated** | Operator/setup errors (misplaced cube, crash) are not policy failures; the assignment requires every attempt to be recorded. |
| **90% Wilson intervals, Beta(s+1, n−s+1) posterior violins, CLD letters from the STEP decision** | Exactly the assignment's reporting format at the 90% level. |
| **Every rollout recorded to `rollout_eval_act40k_te` / `rollout_eval_diff_v2`** | Provides the per-rollout video + data log the submission requires. |

## Known deviations to state in the write-up
- ACT batch 8 vs Diffusion v2 batch 4 → Diffusion saw ~2× fewer training samples.
- Diffusion v1 replaced by v2 before evaluation (v1 too slow for real-time control). v2: 240×320 images, DDIM-10, but U-Net still 512/1024/2048 and batch 4 (~160k samples vs ACT's ~320k).
- Episode 43 (arm idle) is in both policies' training data; episodes 12, 20, 38 (good) are not.
- Matched configs make outcomes correlated (shared scene difficulty); STEP's two-sample model doesn't model this.
- Lighting: eval under recording conditions (blinds closed, lamp on); note any change.

## What to submit (Task 1 folder)
- [ ] Demonstrations: HF dataset `shilpitha-03/so101_cube_grid` + `demos.csv` (all 53 episodes, keep / trained_v1, notes)
- [ ] Evaluated checkpoints + configs: ACT `040000/pretrained_model_te/config.json` + train.sh/train_config; Diffusion `diffusion_v2/config.json` + train_config.json (HF: asma-4a/so101-diffusion-ddim10-ckpt-040000)
- [ ] Logs + videos of every eval rollout: datasets `rollout_eval_act40k_te`, `rollout_eval_diff_v2` (push them to HF)
- [ ] Manifest: `eval_manifest.csv` (episode ID ↔ policy, config, success, partial score, timing, validity)
- [ ] `eval_schedule.csv` (with seed) + `eval_log.csv`
- [ ] Analysis code + outputs: `analyze_eval.py`, `results.md`, `step_result.json`, `violin.png`
- [ ] One-page write-up: question, design choices, results, limitations (deviations above)
- [ ] Reproduction instructions + software versions (LeRobot commit, sequentialized_barnard_tests version from step_result.json)
- [ ] Contribution statement + AI-tool disclosure