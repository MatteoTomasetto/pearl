# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import copy

class SHAC:
    def __init__(self, env, agent, config, manager):

        self.experiment_manager = manager

        # env
        self.env = env

        # agent
        self.actor = agent[0]
        self.critic = agent[1]
        self.target_critic = copy.deepcopy(self.critic)

        # config params
        params = config['trainer_params']
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=float(params["actor_learning_rate"]))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(params["critic_learning_rate"]))
        self.seed = params['seed']
        self.max_steps = params['max_steps']
        self.steps_num = params['short_horizon']
        self.gamma = params['gamma']
        self.rew_scale = params['rew_scale']
        self.critic_method = params['critic_method']
        self.critic_lambda = params['critic_lambda']
        self.critic_iterations = params['critic_iterations']
        self.target_critic_alpha = params['target_critic_alpha']
        self.max_grad_norm = params['max_grad_norm']
        self.grad_clip = params['grad_clip']
        self.device = params["device"]
        self.parametric = params["parametric"]
        self.evaluations = params['evaluations']
        self.save_model = params['save_model']
        self.save_intervals = params['save_intervals']
        self.plot_intervals = params['plot_intervals']
        self.track = params['track']

        # set dimensions 
        self.state_dim = env.Ny
        self.param_dim = env.Np if self.parametric else 0
        self.max_episode_length = env.Nt

        # counters
        self.step_count = 0
        self.episode_reward = 0
        self.episode_len = 0
        self.episode_count = 0

        # buffers
        self.state_buf = torch.zeros((self.steps_num, self.state_dim), dtype=torch.float32, device=self.device)
        self.param_buf = torch.zeros((self.steps_num, self.param_dim), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((self.steps_num, 1), dtype=torch.float32, device=self.device)
        self.done_mask = torch.zeros((self.steps_num, 1), dtype=torch.float32, device=self.device)
        self.next_values = torch.zeros((self.steps_num, 1), dtype=torch.float32, device=self.device)
        self.target_values = torch.zeros((self.steps_num, 1), dtype=torch.float32, device=self.device)


    def compute_actor_loss(self):

        # initialize short-horizon
        state = self.env.initialize_trajectory()
        rew_acc = torch.zeros(1, device=self.device)
        gamma = torch.tensor(1.0, device=self.device)
        next_values = torch.zeros(self.steps_num + 1, device=self.device)
        actor_loss = torch.tensor(0.0, device=self.device)

        # short-horizon rollout
        for i in range(self.steps_num):
            # store data for critic update
            self.state_buf[i] = state.clone().detach().unsqueeze(0)
            if self.parametric:
                self.param_buf[i] = self.env.get_parameters().clone().detach().unsqueeze(0)

            # compute action
            actions = self.actor(state.unsqueeze(0), self.env.get_parameters().unsqueeze(0)) if self.parametric else self.actor(state.unsqueeze(0))

            # apply action            
            state, rew, done, info = self.env.step(torch.tanh(actions))
            self.step_count += 1
            self.episode_len += 1

            # compute reward 
            self.episode_reward += rew.clone().detach().item()
            rew = rew * self.rew_scale
            rew = rew.view(1).to(self.device)

            # compute target critic values
            if not done:
                if self.parametric:
                    next_values[i + 1] = self.target_critic(state.unsqueeze(0), self.env.get_parameters().unsqueeze(0))
                else:
                    next_values[i + 1] = self.target_critic(state.unsqueeze(0))

            if done:
                if torch.isnan(state).sum() > 0 \
                or torch.isinf(state).sum() > 0 \
                or (torch.abs(state) > 1e6).sum() > 0:
                    next_values[i+1] = 0
                
                elif self.episode_len < self.max_episode_length:
                    next_values[i+1] = 0

                else:
                    if self.parametric:
                        next_values[i + 1] = self.target_critic(state.unsqueeze(0), self.env.get_parameters().unsqueeze(0))
                    else:
                        next_values[i + 1] = self.target_critic(state.unsqueeze(0))

            # sanity check
            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                raise ValueError    
        
            # accumulate discounted reward
            rew_acc = rew_acc + gamma * rew
            
            # compute actor loss at the end of the episode/short-horizon
            if i < self.steps_num - 1:
                if done:
                    actor_loss = actor_loss + (-rew_acc - gamma * next_values[i + 1])
            else:
                actor_loss = actor_loss + (-rew_acc - gamma * next_values[i + 1])

            # update gamma
            gamma = gamma * self.gamma

            # episode ends
            if done:
                rew_acc = torch.zeros(1, device=self.device)
                gamma = torch.tensor(1.0, device=self.device)

            # collect data for critic update
            with torch.no_grad():
                self.rew_buf[i] = rew.clone().detach()
                if i < self.steps_num - 1:
                    self.done_mask[i] = done
                else:
                    self.done_mask[i] = 1
                self.next_values[i] = next_values[i + 1].clone()

            # episode ends
            if done:
                self.episode_count += 1
                print(f"Episode {self.episode_count} | Reward: {self.episode_reward:.4f} | Episode Length: {self.episode_len}")

                # logging
                if self.track:
                    self.experiment_manager.log_metrics(metrics={'train/episode_reward': self.episode_reward}, step=self.step_count)

                # rendering
                if (self.env.save_plots or self.env.save_gifs) and (self.episode_count % self.plot_intervals == 0):
                    save_dir = self.experiment_manager.get_media_path(filename=f"episode_{self.episode_count}")
                    self.env.render(save_dir, info=f"episode_{self.episode_count}")

                # checkpointing
                if self.save_model and (self.episode_count % self.save_intervals == 0):
                    save_path = self.experiment_manager.get_model_path(filename=f"episode_{self.episode_count}")
                    self.actor.save(save_path)

                self.episode_reward = 0
                self.episode_len = 0    
                state = self.env.reset()
                state = self.env.initialize_trajectory()

        # loss normalization
        actor_loss = actor_loss / self.steps_num

        return actor_loss

    @torch.no_grad()
    def compute_target_values(self):
        # compute target values
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = torch.zeros(1, device=self.device)
            Bi = torch.zeros(1, device=self.device)
            lam = torch.ones(1, device=self.device)

            for i in reversed(range(self.steps_num)):
                lam = lam * self.critic_lambda * (1. - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (self.critic_lambda * self.gamma * Ai + self.gamma * self.next_values[i] + (1. - lam) / (1. - self.critic_lambda) * self.rew_buf[i])
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.critic_lambda) * Ai + lam * Bi

    def compute_critic_loss(self):
        # compute critic loss 
        predicted = self.critic(self.state_buf, self.param_buf).squeeze(-1) if self.parametric else self.critic(self.state_buf).squeeze(-1)
        target = self.target_values.squeeze(-1)

        return torch.nn.functional.mse_loss(predicted, target)

    def train(self):
        self.env.reset()

        while self.step_count <= self.max_steps:
            # actor update
            self.actor_optimizer.zero_grad()
            actor_loss = self.compute_actor_loss()
            actor_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # online critic update
            self.compute_target_values()
            for _ in range(self.critic_iterations):
                self.critic_optimizer.zero_grad()
                critic_loss = self.compute_critic_loss()
                critic_loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

            # target critic update
            with torch.no_grad():
                for p, p_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                    p_targ.data.mul_(self.target_critic_alpha)
                    p_targ.data.add_((1. - self.target_critic_alpha) * p.data)

            # logging
            if self.track:
                self.experiment_manager.log_metrics(metrics={'critic_loss': critic_loss, 'actor_loss': actor_loss}, step=self.step_count)

        print("Training finished.")

        # checkpointing
        if self.save_model:
            save_path = self.experiment_manager.get_model_path(filename="last")
            self.actor.save(save_path)

        # final evaluation
        self.eval()


    def eval(self):
        # evaluation rollouts
        for i in range(self.evaluations):

            self.episode_reward = 0
            self.episode_len = 0
            done = False

            state = self.env.reset()

            while not done:

                # compute action
                actions = self.actor(state.unsqueeze(0), self.env.get_parameters().unsqueeze(0), eval=True) if self.parametric else self.actor(state.unsqueeze(0), eval=True)

                # apply action
                state, rew, done, info = self.env.step(torch.tanh(actions))
                self.step_count += 1
                self.episode_reward += rew.clone().detach().item()
                self.episode_len += 1

                # episode ends
                if done:
                    print(f"Eval Episode {i} | Reward: {self.episode_reward:.4f} | Episode Length: {self.episode_len}")

                    # logging
                    if self.track:
                        self.experiment_manager.log_metrics(metrics={'eval/episode_reward': self.episode_reward}, step=self.step_count)

                    # rendering 
                    if self.env.save_plots or self.env.save_gifs:
                        save_dir = self.experiment_manager.get_media_path(filename=f"eval_episode_{i}")
                        self.env.render(save_dir, info=f"eval_episode_{i}")
