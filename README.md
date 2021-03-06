# Robot_Autonomy_Project
16-662 Group Project

Members:
Bryson Jones
Quentin Cheng
Shaun Ryer

# Robot Autonomy Spring 2020 Simulation Project

## Installation

Please use Python 3.6

1. Install [PyRep](https://github.com/stepjam/PyRep)
2. Install [RLBench](https://github.com/stepjam/RLBench)
3. `pip install -r requirements.txt`

## Example RLBench Usage
Run `python rlbench_example.py` to launch the example script.
Here, the `BlockPyramid` task is used, and the policy is random end-effector positions.

This script contains example code on how to control the robot, get observations, and get noisy object pose readings.

## Useful Files
The following files may be useful to reference from the In the `rlbench` folder in the `RLBench` repo:
* `rlbench/action_modes.py` - Different action modes to control the robot
* `rlbench/backend/observation.py` - All fields available in the observation object