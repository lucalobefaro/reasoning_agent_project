# reasoning_agent_project

# Usage

To run experiments, execute the script `ac_sapientino_case.py` with as argument the directory in which the experiment config and results should be stored. The config file should already be present in the directory as a file `params.cfg`; see `experiment_template/params.cfg` for an example.

Additional options are available:

- `--episodes N` (default 300): controls the number `N` of episodes to run

- `--render_interval N` (default 5): each `N` episodes, an episode will be rendered and some logs displayed (e.g. recent avg reward) to monitor the training

