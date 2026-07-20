
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import torch
import os

class ExperimentManager:
    def __init__(self, config: dict, seed):
        self.config = config
        self.global_cfg = config.get('global', {})
        self.env_cfg = config.get('env', {})
        self.logger_cfg = config.get('logger', {})
        self.agent_cfg = config.get('agent', {})
        self.trainer_cfg = config.get('trainer', {})

        # Env
        self.env_type = self.env_cfg.get('env_type')

        # Logger
        self.logger_type = self.logger_cfg.get('logger_type')

        # Agent
        self.agent_type = self.agent_cfg.get('agent_type')

        # Trainer
        self.trainer_type = self.trainer_cfg.get('trainer_type')
        
        # Global settings
        self.seed = seed
        self.env_cfg["env_params"]["seed"] = self.seed
        self.agent_cfg["agent_params"]["seed"] = self.seed
        self.trainer_cfg["trainer_params"]["seed"] = self.seed

        self.agent_cfg["agent_params"]["device"] = self.global_cfg['device']
        self.env_cfg["env_params"]["device"] = self.global_cfg['device']
        
        self.track = self.logger_cfg.get('track')
        self.trainer_cfg["trainer_params"]["track"] = self.track
        self.trainer_cfg["trainer_params"]["device"] = self.global_cfg['device']
        self.trainer_cfg["trainer_params"]["parametric"] = self.agent_cfg["agent_params"]["parametric"]

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"{self.global_cfg.get('exp_name')}_seed_{self.seed}__t_{self.timestamp}"
        
        # Define the base directory for this specific run
        self.exp_dir = Path("experiments") / self.exp_name
        
        # Sub-directories for different artifacts
        self.model_dir = self.exp_dir / "models"
        self.media_dir = self.exp_dir / "media"
        self.log_dir = self.exp_dir / "logs"
        
        self._setup_folders()
        self._save_config_backup()


    def _setup_folders(self):
        """Create the directory structure on disk."""
        for folder in [self.model_dir, self.media_dir, self.log_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def _save_config_backup(self):
        """Keep a copy of the config inside the experiment folder."""
        import yaml
        with open(self.exp_dir / "config_backup.yaml", 'w') as f:
            yaml.dump(self.config, f)

    def _build_agent(self, state_dim, action_dim, adjoint_dim, param_dim):
        """Builder for the agent."""
        agent_params = self.agent_cfg.get('agent_params')
             
        if self.agent_type == "bptt_agent":
            from agents.agents import Actor
            raw_actor = Actor(state_dim, action_dim, param_dim, agent_params)
            return raw_actor
              
        elif self.agent_type == "shac_agent":
            from agents.agents import Actor, Critic
            raw_actor = Actor(state_dim, action_dim, param_dim, agent_params)
            raw_critic = Critic(state_dim, param_dim, agent_params)
            return [raw_actor, raw_critic]
        
        elif self.agent_type == "pearl_agent":
            from agents.agents import Actor, AdjointNet
            raw_actor = Actor(state_dim, action_dim, param_dim, agent_params)
            raw_critic = AdjointNet(state_dim, adjoint_dim, param_dim, agent_params)
            return [raw_actor, raw_critic]
        
    def _build_env(self):
        """Builder for the env"""
        env_params = self.env_cfg.get('env_params')
        
        if self.env_type == "leaderfollower_singleagent":
            from envs.leaderfollower import LeaderFollower_SingleAgent
            raw_env = LeaderFollower_SingleAgent(env_params)
            return raw_env
        
        elif self.env_type == "leaderfollower_meanfield":
            from envs.leaderfollower import LeaderFollower_MeanField
            raw_env = LeaderFollower_MeanField(env_params)
            return raw_env
               
        else:
            raise ValueError(f"Env {self.env_type} not supported.")
    
    def _init_logger(self):
        """Initialize the logger"""
        logger_params = self.logger_cfg.get('logger_params')
        mode = self.logger_cfg.get('mode')

        if self.track:
            if self.logger_type == 'wandb':
                import wandb
                project_name = logger_params.get('project_name')

                run = wandb.init(
                    mode=mode,
                    project=project_name,
                    name=self.exp_name,
                    settings={
                    "_service_wait": 600,
                    "init_timeout": 600
                    }
                )
                self.run_dir = Path(run.dir).parent

                return wandb
    
    def _set_seed(self):
        """"Set the global seed"""
        # Base setup
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
    def instantiate(self):
        """Instantiate the Trainer"""
        # Set the seed
        self._set_seed()

        # Build the env
        self.env = self._build_env()
        state_dim = self.env.Ny
        action_dim = self.env.Nu
        adjoint_dim =  self.env.Ny
        param_dim = self.env.Np if self.agent_cfg["agent_params"]["parametric"] else 0

        # Build the agent
        self.agent = self._build_agent(state_dim, action_dim, adjoint_dim, param_dim)
        
        # Initialize the logger
        self.logger = self._init_logger()

        # Create the trainer
        if self.trainer_type == "bptt_trainer":
            from trainers.bptt_trainer import BPTT
            trainer = BPTT(                
                env=self.env, 
                agent=self.agent, 
                config=self.trainer_cfg, 
                manager=self
                )

        elif self.trainer_type == "shac_trainer":
            from trainers.shac_trainer import SHAC
            trainer = SHAC(
                env=self.env,
                agent=self.agent,
                config=self.trainer_cfg,
                manager=self
                )
        
        elif self.trainer_type == "pearl_trainer":
            from trainers.pearl_trainer import PEARL
            trainer = PEARL(
                env=self.env,
                agent=self.agent,
                config=self.trainer_cfg,
                manager=self
                )

        else:
            raise ValueError(f"Env {self.trainer_type} not supported.")
        
        return trainer
        
    def log_metrics(self, metrics: dict, step: int):
        """Use by the trainer to log metrics on the selected logger"""

        if self.logger_type  == 'wandb':
            self.logger.log(metrics, step=step)
        else:
            for k, v in metrics.items():
                self.logger.add_scalar(k, v, step)
    
    def get_model_path(self, filename: str) -> str:
        """Returns the full local path to save a model file."""
        return str(self.model_dir / filename)

    def get_media_path(self, filename: str) -> str:
        """Returns the full local path to save an image or GIF."""
        return str(self.media_dir / filename)

