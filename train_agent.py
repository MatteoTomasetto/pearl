from managers.experiment_manager import ExperimentManager
import yaml
import wandb
import argparse
import subprocess
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-type', type=str, default='pearl',
                        help='Agent type (bptt, shac, pearl).')
    parser.add_argument('--experiment-type', type=str, default='leaderfollower_singleagent',
                        help='Experiment type (leaderfollower_singleagent, leaderfollower_meanfield).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed.')
    args = parser.parse_args()

    seed = args.seed
    agent_type = args.agent_type
    experiment_type = args.experiment_type

    with open("configs/{}_{}.yaml".format(agent_type, experiment_type), "r") as f:
        config_dict = yaml.safe_load(f)
        manager = ExperimentManager(config_dict, seed)
        trainer = manager.instantiate()
        trainer.train()

    try:
        dir = manager.run_dir
        wandb.finish()

        subprocess.run(
            ["wandb", "sync", str(dir)],
            check=False
        )

    except:
        pass
