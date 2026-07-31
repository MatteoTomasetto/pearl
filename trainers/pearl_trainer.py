import torch
import copy

class PEARL:
    def __init__(self, env, agent, config, manager):

        self.experiment_manager = manager

        # env
        self.env = env

        # agent
        self.actor = agent[0]
        self.adjoint_net = agent[1]
        self.target_adjoint_net = copy.deepcopy(self.adjoint_net)

        # config params
        params = config['trainer_params']
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=float(params["actor_learning_rate"]))
        self.adjoint_net_optimizer = torch.optim.Adam(self.adjoint_net.parameters(), lr=float(params["adjoint_net_learning_rate"]))
        self.seed = params['seed']
        self.max_steps = params['max_steps']
        self.steps_num = params['short_horizon']
        self.gamma = params['gamma']
        self.adjoint_net_lambda = params['adjoint_net_lambda']
        self.adjoint_net_iterations = params['adjoint_net_iterations']
        self.target_adjoint_net_alpha = params['target_adjoint_net_alpha']
        self.max_grad_norm = params['max_grad_norm']
        self.grad_clip = params['grad_clip']
        self.device = params["device"]
        self.parametric = params["parametric"]
        self.evaluations = params['evaluations']
        self.save_model = params['save_model']
        self.save_intervals = params['save_intervals']
        self.plot_intervals = params['plot_intervals']
        self.track = params['track']

        # counters
        self.step_count = 0
        self.episode_reward = 0
        self.episode_len = 0
        self.episode_count = 0

        # buffers
        self.state_buf = []
        self.param_buf = []
        self.gammas = []
        self.rew_buf = []
        self.adjoints = []
        self.target_adjoints = []
       
        # utils
        self.done = False
        self.steps_done = self.steps_num

    def compute_actor_loss(self):

        # initialize short-horizon
        state = self.env.initialize_trajectory()
        rew_acc = torch.zeros(1, device=self.device)
        gamma = torch.tensor(1.0, device=self.device)        
        actor_loss = torch.tensor(0.0, device=self.device)
        self.done = False
        self.steps_done = torch.tensor(self.steps_num, device=self.device)

        # short-horizon rollout
        for i in range(self.steps_num):
            # store data for adjoint net update
            self.state_buf.append(state)
            self.gammas.append(gamma)
            if self.parametric:
                self.param_buf.append(self.env.get_parameters().clone().detach())

            # compute action
            actions = self.actor(state.unsqueeze(0), self.env.get_parameters().unsqueeze(0)) if self.parametric else self.actor(state.unsqueeze(0))

            # apply action
            state, rew, self.done, info = self.env.step(torch.tanh(actions))
            self.step_count += 1
            self.episode_len += 1

            # compute reward 
            self.episode_reward += rew.clone().detach().item()
            rew = rew.view(1).to(self.device)

            # accumulate discounted reward
            rew_acc = rew_acc + gamma * rew
            self.rew_buf.append(gamma * rew)

            # update gamma
            gamma = gamma * self.gamma

            # compute actor loss at the end of the episode/short-horizon
            if self.done or i == self.steps_num - 1:
                actor_loss = -rew_acc

                self.state_buf.append(state)
                self.gammas.append(gamma)
                if self.parametric:
                    self.param_buf.append(self.env.get_parameters())

            # episode ends
            if self.done:

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

                self.env.reset()
                self.episode_reward = 0
                self.episode_len = 0
                self.steps_done = torch.tensor(i+1, device=self.device)

                break
        
        # loss normalization
        actor_loss = actor_loss / self.steps_done
        for i in range(len(self.rew_buf)):
            self.rew_buf[i] = self.rew_buf[i] / self.steps_done
        
        return actor_loss
    
    @torch.no_grad()
    def compute_adjoints(self):
        # compute target adjoints with TD-lambda
        self.target_adjoints.append(torch.zeros(self.env.Ny, device=self.device))

        for i in reversed(range(1, len(self.state_buf))):
            
            if self.parametric:
                    self.adjoints.insert(0, self.gammas[i] * self.target_adjoint_net(self.state_buf[i].detach().unsqueeze(0), self.param_buf[i].detach().unsqueeze(0))[0] / self.steps_done)
            else:
                    self.adjoints.insert(0, self.gammas[i] * self.target_adjoint_net(self.state_buf[i].detach().unsqueeze(0))[0] / self.steps_done)

            grad_outputs = (self.adjoint_net_lambda * self.target_adjoints[0] + (1 - self.adjoint_net_lambda) * self.adjoints[0])
            self.target_adjoints.insert(0, torch.autograd.grad(self.state_buf[i], self.state_buf[i-1], grad_outputs = grad_outputs, retain_graph=True, create_graph=False)[0] - torch.autograd.grad(self.rew_buf[i-1], self.state_buf[i-1], retain_graph=True, create_graph=False)[0])

    def compute_adjoint_net_loss(self):
        # compute online adjoint net loss
        if self.parametric:
            predicted = self.adjoint_net(torch.stack(self.state_buf).detach(), torch.stack(self.param_buf).detach()) * torch.stack(self.gammas).unsqueeze(1) / self.steps_done
        else:
            predicted = self.adjoint_net(torch.stack(self.state_buf).detach()) * torch.stack(self.gammas).unsqueeze(1) / self.steps_done

        target = torch.stack(self.target_adjoints)

        if not self.done:
            predicted = predicted[:-1]
            target = target[:-1]

        return torch.nn.functional.mse_loss(predicted, target)

    def update_adjoint_net(self):
        # update online adjoint net
        for _ in range(self.adjoint_net_iterations):
            self.adjoint_net_optimizer.zero_grad()
            adjoint_net_loss = self.compute_adjoint_net_loss()
            adjoint_net_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.adjoint_net.parameters(), self.max_grad_norm)
            self.adjoint_net_optimizer.step()
        
        return adjoint_net_loss

    @torch.no_grad()
    def update_target_adjoint_net(self):
        # soft update for target adjoint net
        for p, p_targ in zip(self.adjoint_net.parameters(), self.target_adjoint_net.parameters()):
            p_targ.data.mul_(self.target_adjoint_net_alpha)
            p_targ.data.add_((1. - self.target_adjoint_net_alpha) * p.data)

    def train(self):
        self.env.reset()
        self.state_buf = []
        self.param_buf = []
        self.gammas = []
        self.rew_buf = []
        self.adjoints = []
        self.target_adjoints = []

        while self.step_count <= self.max_steps:
            # compute actor loss
            self.actor_optimizer.zero_grad()
            actor_loss = self.compute_actor_loss()

            # compute adjoints
            self.compute_adjoints()

            # correct terminal adjoint in the tape
            if not self.done:    
                self.state_buf[-1].register_hook(lambda g, adj = self.adjoints[-1]: g + adj) # override final adjoint manually

            # actor update
            actor_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            try:
                self.env._env.detach()
            except:
                pass

            # online adjoint net update
            adjoint_net_loss = self.update_adjoint_net()

            # target adjoint net update
            self.update_target_adjoint_net()

            # logging
            if self.track:
                self.experiment_manager.log_metrics(metrics={'adjoint_net_loss': adjoint_net_loss, 'actor_loss': actor_loss}, step=self.step_count)
                
            self.state_buf = []
            self.param_buf = []
            self.gammas = []
            self.rew_buf = []
            self.adjoints = []
            self.target_adjoints = []
        
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
