# Task 1 evaluation results

## Per-policy summary

| policy                               |   trials |   successes |   failures | success_rate   | wilson90   |   completion_median_s | completion_IQR_s   |   n_timed |   mean_progress | seen   | unseen   |   stage_0.0 |   stage_0.25 |   stage_0.5 |   stage_0.75 |   stage_1.0 |
|:-------------------------------------|---------:|------------:|-----------:|:---------------|:-----------|----------------------:|:-------------------|----------:|----------------:|:-------|:---------|------------:|-------------:|------------:|-------------:|------------:|
| ACT (40k, temporal ensembling)       |       25 |          14 |         11 | 56%            | [40%, 71%] |                  22   | 19.7–22.9          |        13 |            0.72 | 9/16   | 5/9      |           0 |            6 |           5 |            0 |          14 |
| Diffusion v2 (40k, 240x320, DDIM-10) |       25 |           9 |         16 | 36%            | [22%, 52%] |                  19.6 | 17.5–21.8          |         8 |            0.54 | 6/16   | 3/9      |           0 |           15 |           0 |            1 |           9 |

stage_x = number of trials whose furthest stage was x (0 none, 0.25 reached, 0.5 grasped, 0.75 transported, 1 placed).
Completion time = policy start → gripper release over the tray (from recorded gripper command); failures ran to the 60 s timeout.

## STEP

- **decision**: FailToDecide
- **meaning**: inconclusive: difference not established (NOT evidence that they are equal)
- **first_decision_trials_per_policy**: None
- **n_used**: 25
- **n_max**: 25
- **alpha_pair**: 0.1
- **alternative**: P0LessThanP1 (A=ACT, B=Diffusion)
- **test**: MirroredStepTest(inference_mode='ranking'), default linear risk budget
- **package_version**: 0.0.9

## Failures (from operator notes)

|   trial | grid_id   |   seen | policy    |   score | note                                                                                                                                                                                                                     |
|--------:|:----------|-------:|:----------|--------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|       1 | B1        |      1 | diffusion |    0.25 | appraoched and drifted away couldnt recover                                                                                                                                                                              |
|       2 | D4        |      1 | act       |    0.25 | gripped it at the edge; flew off couldnt recover                                                                                                                                                                         |
|       3 | B2        |      1 | diffusion |    0.25 | gripped on the edge; flew away couldnt recover                                                                                                                                                                           |
|       4 | C4        |      1 | act       |    0.25 | failed grasp; cube moved; shadow on cube and could not recover                                                                                                                                                           |
|       6 | A1        |      1 | diffusion |    0.25 | Could not get the grasp right                                                                                                                                                                                            |
|       8 | c-A3      |      0 | act       |    0.25 | it stopped and couldnt grasp even though it was a good grasp position                                                                                                                                                    |
|       9 | c-B3      |      0 | act       |    0.25 | couldnt grasp; went really close tho                                                                                                                                                                                     |
|      10 | C3        |      1 | diffusion |    0.25 | aligned the jaw but failed to close the gripper; it was on the top edge of the cube; maybe cause of the tilt? then failed to adjust cause the cube turned                                                                |
|      11 | c-C1      |      0 | diffusion |    0.25 | it went on top of the cube moved around and tried to grip in the air to the right of the cube                                                                                                                            |
|      11 | c-C1      |      0 | act       |    0.5  | grasped the cube but it dint move over at all                                                                                                                                                                            |
|      13 | c-B1      |      0 | diffusion |    0.25 | grasp kept trying and it just kept missing couldnt align the gripper well enough to grasp                                                                                                                                |
|      14 | c-C2      |      0 | diffusion |    0.25 | went to the right kept trying to grasp in air                                                                                                                                                                            |
|      15 | C1        |      1 | diffusion |    0.25 | again ended up on the cubes right and tried to keep moving to the left and couldnt reach                                                                                                                                 |
|      15 | C1        |      1 | act       |    0.5  | gripped too hard and made the cube fly away ended up in a position where we couldnt see it in any view                                                                                                                   |
|      16 | D3        |      1 | diffusion |    0.25 | it ended up on cubes right and trying to grasp in air                                                                                                                                                                    |
|      17 | B4        |      1 | act       |    0.25 | hovered over it almost gripping it but just dint close all the way noticed the cube wasnt visible in the top view anymore                                                                                                |
|      19 | D1        |      1 | act       |    0.25 | was on the right edge and couldnt grasp even after multiple attempts to close the gripper                                                                                                                                |
|      19 | D1        |      1 | diffusion |    0.25 | it was trying to grasp to the right of the cube in air                                                                                                                                                                   |
|      20 | c-A2      |      0 | diffusion |    0.25 | gripped on the edge sent the cube flying but this time it was in frame of both the cameras but it ended up in a place it hasnt learnt to pick from but still went towards it might have been outside reachable workspace |
|      21 | D2        |      1 | act       |    0.5  | picked it up but couldnt move it to the tray                                                                                                                                                                             |
|      21 | D2        |      1 | diffusion |    0.25 | tried to grasp to the right of the cube in air                                                                                                                                                                           |
|      22 | A4        |      1 | diffusion |    0.75 | dropped outside the tray and tried to reach to pick it up again but never seen how to pick in that position                                                                                                              |
|      22 | A4        |      1 | act       |    0.5  | missed it moved it tried to reach again and failed                                                                                                                                                                       |
|      23 | c-C3      |      0 | act       |    0.5  | gripped flew awayy as it was moving to the tray so maybe the actual score is 0.75 but could recover and pick it up again                                                                                                 |
|      23 | c-C3      |      0 | diffusion |    0.25 | gripped flew away couldnt recover                                                                                                                                                                                        |
|      24 | c-A1      |      0 | diffusion |    0.25 | turned the cube as it tried to pick it up and couldnt recover                                                                                                                                                            |
|      25 | C2        |      1 | diffusion |    0.25 | tried grasping on the right of the cube                                                                                                                                                                                  |

## Invalid (repeated) runs: 0

none
