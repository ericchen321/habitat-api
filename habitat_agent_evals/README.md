agent_evals
==============================
### Installation

The `habitat_agent_evals` is included upon installation by default. 


### Summary
The sub-package includes the following scripts:
* `eval_agent_discrete.py`: evaluate a Habitat agent that produces discrete actions to Habitat Sim on point-goal navigation episodes. Reports average metrics (distance to goal, SPL, success) over the episodes;
* `eval_agent_continuous.py`: evaluate a Habitat agent that produces continuous velocities to Habitat Sim (with object dynamics simulation enabled) on point-goal navigation episodes.  Reports average metrics (distance to goal, SPL, success) over the episodes
* `remap_baseline_weights.py`: remaps Habitat agent from old format (prior to 2020) to new the new format that is compatible with our Habitat API. [The script was created by Erik Wijmans](https://gist.github.com/erikwijmans/94f33461982ac9e3cc354fb26e4b80e1)


### Evaluate Discrete Agent
Run `eval_agent_discrete.py` and specify evaluation parameters such as agent's sensor type, agent's model, task configurationm etc. Example:
```bash
python habitat_agent_evals/eval_agent_discrete.py --input-type depth --model-path data/checkpoints/depth_new.pth --task-config configs/tasks/pointnav.yaml --num-episodes=50 --frame-rate=10
```

### Evaluate Continuous Agent
Run `eval_agent_continuous.py` and specify evaluation parameters such as agent's sensor type, agent's model, task configurationm etc. Example:
```bash
habitat_agent_evals/eval_agent_continuous.py --input-type depth --model-path data/checkpoints/depth_new.pth --task-config configs/tasks/pointnav.yaml --num-episodes 50 --frame-rate=10 --control-period 1.0 
```

**TensorBoard and Video Generation Support**
You can check the sensor inputs from an agent, as well as a top-down map showing the agent's navigation steps by running TensorBoard:
```bash
tensorboard --logdir=tb_benchmark_dir/
```
You can also check navigation episode recordings under `video_benchmark_dir/`. A generated video should look like [this](https://drive.google.com/file/d/1fmbW5vny-nmmG6aToTYmU5zj1eeAV4u0/view?usp=sharing).
