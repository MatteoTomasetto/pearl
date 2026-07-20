import torch

class BPTT:
    def __init__(self, env, agent, config, manager):

        self.experiment_manager = manager

        self.env = env
        self.actor = agent

        params = config["trainer_params"]
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=float(params["actor_learning_rate"]))

        self.seed = params["seed"]     
        self.max_steps = params["max_steps"]
        self.steps_num = params['short_horizon']
        self.gamma = params["gamma"]
        self.max_grad_norm = params['max_grad_norm']
        self.grad_clip = params['grad_clip']
        self.device = params["device"]
        self.parametric = params["parametric"]
        self.evaluations = params['evaluations']
        self.save_model = params['save_model']
        self.save_intervals = params['save_intervals']
        self.plot_intervals = params['plot_intervals']
        self.track = params['track']

        # for logging
        self.step_count = 0
        self.episode_reward = 0
        self.episode_len = 0
        self.episode_count = 0

        # utils
        self.steps_done = self.steps_num

    def compute_actor_loss(self):
        rew_acc = torch.zeros(1,  device=self.device)
        gamma = torch.tensor(1.0, device = self.device)
       
        state = self.env.initialize_trajectory()
       
        actor_loss = torch.tensor(0.0, device=self.device)

        self.steps_done = torch.tensor(self.steps_num, device=self.device)

        for i in range(self.steps_num):
            # action
            actions = self.actor(state.unsqueeze(0), self.env.get_parameters().unsqueeze(0)) if self.parametric else self.actor(state.unsqueeze(0))
            state, rew, done, info = self.env.step(torch.tanh(actions))

            self.step_count += 1
            self.episode_len += 1
            self.episode_reward += rew

            # raw reward 
            rew = rew.view(1).to(self.device)

            # accumulate discounted reward
            rew_acc = rew_acc + gamma * rew

            gamma = gamma * self.gamma

            if done or i == self.steps_num - 1:
                actor_loss = -rew_acc

            if done:

                self.episode_count += 1

                print(f"Episode {self.episode_count} | Reward: {self.episode_reward:.4f} | Episode Length: {self.episode_len}")

                # Logging
                if self.track:
                    self.experiment_manager.log_metrics(metrics={'train/episode_reward': self.episode_reward}, step=self.step_count)

                # Rendering
                if (self.env.save_plots or self.env.save_gifs) and (self.episode_count % self.plot_intervals == 0):
                    save_dir = self.experiment_manager.get_media_path(filename=f"episode_{self.episode_count}")
                    self.env.render(save_dir, info=f"episode_{self.episode_count}")

                # Checkpointing
                if self.save_model and (self.episode_count % self.save_intervals == 0):
                    save_path = self.experiment_manager.get_model_path(filename=f"episode_{self.episode_count}")
                    self.actor.save(save_path)

                self.env.reset()

                self.episode_reward = 0
                self.episode_len = 0

                self.steps_done = torch.tensor(i+1, device=self.device)

                break
        
        # average actor loss over steps
        actor_loss = actor_loss / self.steps_done

        return actor_loss

    def train(self):
        self.env.reset()

        while self.step_count <= self.max_steps:

            # === ACTOR UPDATE ===
            self.actor_optimizer.zero_grad()
            actor_loss = self.compute_actor_loss()
            actor_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            try:
                self.env._env.detach()
            except:
                pass

            if self.track:
                self.experiment_manager.log_metrics(metrics={'actor_loss': actor_loss}, step=self.step_count)

        print("Training finished.")

        # Checkpointing
        if self.save_model:
            save_path = self.experiment_manager.get_model_path(filename="last")
            self.actor.save(save_path)

        self.eval()

    def eval(self):

        for i in range(self.evaluations):

            self.episode_reward = 0
            self.episode_len = 0
            done = False

            state = self.env.reset()

            while not done:

                # action
                actions = self.actor(state.unsqueeze(0), self.env.get_parameters().unsqueeze(0), eval=True) if self.parametric else self.actor(state.unsqueeze(0), eval=True)

                state, rew, done, info = self.env.step(torch.tanh(actions))
                
                self.step_count += 1
                self.episode_reward += rew.clone().detach().item()
                self.episode_len += 1

                if done:
                    print(f"Eval Episode {i} | Reward: {self.episode_reward:.4f} | Episode Length: {self.episode_len}")

                    if self.track:
                        self.experiment_manager.log_metrics(metrics={'eval/episode_reward': self.episode_reward}, step=self.step_count)

                    if self.env.save_plots or self.env.save_gifs:
                        save_dir = self.experiment_manager.get_media_path(filename=f"eval_episode_{i}")
                        self.env.render(save_dir, info=f"eval_episode_{i}")
