import os
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def get_activation_func(activation_name):
    if activation_name.lower() == 'tanh':
        return nn.Tanh()
    elif activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'elu':
        return nn.ELU()
    elif activation_name.lower() == 'identity':
        return nn.Identity()
    else:
        raise NotImplementedError('Actication function {} not defined'.format(activation_name))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, param_dim, cfg_network):
        super(Actor, self).__init__()

        self.device = cfg_network["device"]
        self.deterministic = cfg_network["deterministic"]
        self.parametric = cfg_network["parametric"]
        self.param_dim = param_dim

        self.last_activation = cfg_network["last_activation"]

        if self.parametric:
            state_layer_dims = [state_dim] + cfg_network['actor']['state_units']
            state_modules = []
            for i in range(len(state_layer_dims) - 1):
                state_modules.append(nn.Linear(state_layer_dims[i], state_layer_dims[i + 1]))
                if i < len(state_layer_dims) - 2:
                    state_modules.append(get_activation_func(cfg_network['actor']['activation']))
                    state_modules.append(torch.nn.LayerNorm(state_layer_dims[i + 1]))
                else:
                    state_modules.append(get_activation_func('identity'))

            if len(state_modules) == 0:
                state_modules.append(get_activation_func('identity'))

            self.state_net = nn.Sequential(*state_modules).to(self.device)

            param_layer_dims = [param_dim] + cfg_network['actor']['param_units']
            param_modules = []
            for i in range(len(param_layer_dims) - 1):
                param_modules.append(nn.Linear(param_layer_dims[i], param_layer_dims[i + 1]))
                if i < len(param_layer_dims) - 2:
                    param_modules.append(get_activation_func(cfg_network['actor']['activation']))
                    param_modules.append(torch.nn.LayerNorm(param_layer_dims[i + 1]))
                else:
                    param_modules.append(get_activation_func('identity'))

            if len(param_modules) == 0:
                param_modules.append(get_activation_func('identity'))

            self.param_net = nn.Sequential(*param_modules).to(self.device)

            head_layer_dims = [state_layer_dims[-1] + param_layer_dims[-1]] + cfg_network['actor']['head_units'] + [action_dim]
            head_modules = []
            for i in range(len(head_layer_dims) - 1):
                head_modules.append(nn.Linear(head_layer_dims[i], head_layer_dims[i + 1]))
                if i < len(head_layer_dims) - 2:
                    head_modules.append(get_activation_func(cfg_network['actor']['activation']))
                    head_modules.append(torch.nn.LayerNorm(head_layer_dims[i + 1]))
                else:
                    if self.last_activation:
                        head_modules.append(get_activation_func('tanh'))
                    else:
                        head_modules.append(get_activation_func('identity'))

            self.actor = nn.Sequential(*head_modules).to(self.device)

        else:
            layer_dims = [state_dim] + cfg_network['actor']['units'] + [action_dim]

            modules = []
            for i in range(len(layer_dims) - 1):
                modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                if i < len(layer_dims) - 2:
                    modules.append(get_activation_func(cfg_network['actor']['activation']))
                    modules.append(torch.nn.LayerNorm(layer_dims[i + 1]))
                else:
                    if self.last_activation:
                        modules.append(get_activation_func('tanh'))
                    else:
                        modules.append(get_activation_func('identity'))

            self.actor = nn.Sequential(*modules).to(self.device)

        logstd = cfg_network.get('actor_logstd_init', -1.0)
        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=self.device) * logstd)

        # Zero init for the last layer
        for module in reversed(self.actor):
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)
                nn.init.zeros_(module.bias)
                break

    def forward(self, state, param=None, eval=False):
        if self.parametric:
            state_encoding = self.state_net(state.to(self.device))
            param_encoding = self.param_net(param.to(self.device))
            state_param_encodings = torch.cat([state_encoding, param_encoding], dim=-1)
            action = self.actor(state_param_encodings)
        else:
            action = self.actor(state.to(self.device))

        if self.deterministic or eval:
            return action
        else:
            std = self.logstd.exp() 
            self.dist = Normal(action, std)
            sample = self.dist.rsample()
            return sample

    def save(self, directory, episode_num=0, filename="policy"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        full_filename = filename + '_ep_' + str(episode_num)
        full_path = os.path.join(directory, "{}.pt".format(full_filename))
        torch.save(self.state_dict(), full_path)
        print(f"Policy saved at {full_path}")

    def load(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint)
        self.to(device)
        print(f"Policy loaded from {path}")


class Critic(nn.Module):
    def __init__(self, state_dim, param_dim, cfg_network):
        super(Critic, self).__init__()

        self.device = cfg_network["device"]
        self.parametric = cfg_network["parametric"]
        self.param_dim = param_dim

        if self.parametric:
            state_layer_dims = [state_dim] + cfg_network['critic']['state_units']
            state_modules = []
            for i in range(len(state_layer_dims) - 1):
                state_modules.append(nn.Linear(state_layer_dims[i], state_layer_dims[i + 1]))
                if i < len(state_layer_dims) - 2:
                    state_modules.append(get_activation_func(cfg_network['critic']['activation']))
                    state_modules.append(torch.nn.LayerNorm(state_layer_dims[i + 1]))
                else:
                    state_modules.append(get_activation_func('identity'))

            if len(state_modules) == 0:
                state_modules.append(get_activation_func('identity'))

            self.state_net = nn.Sequential(*state_modules).to(self.device)

            param_layer_dims = [param_dim] + cfg_network['critic']['param_units']
            param_modules = []
            for i in range(len(param_layer_dims) - 1):
                param_modules.append(nn.Linear(param_layer_dims[i], param_layer_dims[i + 1]))
                if i < len(param_layer_dims) - 2:
                    param_modules.append(get_activation_func(cfg_network['critic']['activation']))
                    param_modules.append(torch.nn.LayerNorm(param_layer_dims[i + 1]))
                else:
                    param_modules.append(get_activation_func('identity'))

            if len(param_modules) == 0:
                param_modules.append(get_activation_func('identity'))

            self.param_net = nn.Sequential(*param_modules).to(self.device)

            head_layer_dims = [state_layer_dims[-1] + param_layer_dims[-1]] + cfg_network['critic']['head_units'] + [1]
            head_modules = []
            for i in range(len(head_layer_dims) - 1):
                head_modules.append(nn.Linear(head_layer_dims[i], head_layer_dims[i + 1]))
                if i < len(head_layer_dims) - 2:
                    head_modules.append(get_activation_func(cfg_network['critic']['activation']))
                    head_modules.append(torch.nn.LayerNorm(head_layer_dims[i + 1]))
                else:
                    head_modules.append(get_activation_func('identity'))

            self.critic = nn.Sequential(*head_modules).to(self.device)
        else:
            layer_dims = [state_dim] + cfg_network['critic']['units'] + [1]

            modules = []
            for i in range(len(layer_dims) - 1):
                modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                if i < len(layer_dims) - 2:
                    modules.append(get_activation_func(cfg_network['critic']['activation']))
                    modules.append(torch.nn.LayerNorm(layer_dims[i + 1]))
                else:
                    modules.append(get_activation_func('identity'))

            self.critic = nn.Sequential(*modules).to(self.device)

        # Zero init for the last layer
        for module in reversed(self.critic):
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)
                nn.init.zeros_(module.bias)
                break

    def forward(self, state, param=None):
        if self.parametric:
            state_encoding = self.state_net(state.to(self.device))
            param_encoding = self.param_net(param.to(self.device))
            state_param_encodings = torch.cat([state_encoding, param_encoding], dim=-1)
            value = self.critic(state_param_encodings)
        else:
            value = self.critic(state.to(self.device))

        return value

    def save(self, directory, episode_num=0, filename="critic"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        full_filename = filename + '_ep_' + str(episode_num)
        full_path = os.path.join(directory, "{}.pt".format(full_filename))
        torch.save(self.state_dict(), full_path)
        print(f"Critic saved at {full_path}")

    def load(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint)
        self.to(device)
        print(f"Critic loaded from {path}")


class AdjointNet(nn.Module):
    def __init__(self, state_dim, adjoint_dim, param_dim, cfg_network):
        super(AdjointNet, self).__init__()

        self.device = cfg_network["device"] 
        self.parametric = cfg_network["parametric"]
        self.param_dim = param_dim

        if self.parametric:
            state_layer_dims = [state_dim] + cfg_network['adjoint_net']['state_units']
            state_modules = []
            for i in range(len(state_layer_dims) - 1):
                state_modules.append(nn.Linear(state_layer_dims[i], state_layer_dims[i + 1]))
                if i < len(state_layer_dims) - 2:
                    state_modules.append(get_activation_func(cfg_network['adjoint_net']['activation']))
                    state_modules.append(torch.nn.LayerNorm(state_layer_dims[i + 1]))
                else:
                    state_modules.append(get_activation_func('identity'))

            if len(state_modules) == 0:
                state_modules.append(get_activation_func('identity'))

            self.state_net = nn.Sequential(*state_modules).to(self.device)

            param_layer_dims = [param_dim] + cfg_network['adjoint_net']['param_units']
            param_modules = []
            for i in range(len(param_layer_dims) - 1):
                param_modules.append(nn.Linear(param_layer_dims[i], param_layer_dims[i + 1]))
                if i < len(param_layer_dims) - 2:
                    param_modules.append(get_activation_func(cfg_network['adjoint_net']['activation']))
                    param_modules.append(torch.nn.LayerNorm(param_layer_dims[i + 1]))
                else:
                    param_modules.append(get_activation_func('identity'))

            if len(param_modules) == 0:
                param_modules.append(get_activation_func('identity'))

            self.param_net = nn.Sequential(*param_modules).to(self.device)

            head_layer_dims = [state_layer_dims[-1] + param_layer_dims[-1]] + cfg_network['adjoint_net']['head_units'] + [adjoint_dim]
            head_modules = []
            for i in range(len(head_layer_dims) - 1):
                head_modules.append(nn.Linear(head_layer_dims[i], head_layer_dims[i + 1]))
                if i < len(head_layer_dims) - 2:
                    head_modules.append(get_activation_func(cfg_network['adjoint_net']['activation']))
                    head_modules.append(torch.nn.LayerNorm(head_layer_dims[i + 1]))
                else:
                    head_modules.append(get_activation_func('identity'))

            self.adjoint_net = nn.Sequential(*head_modules).to(self.device)
        else:
            self.layer_dims = [state_dim] + cfg_network['adjoint_net']['units'] + [adjoint_dim]

            modules = []
            for i in range(len(self.layer_dims) - 1):
                modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
                if i < len(self.layer_dims) - 2:
                    modules.append(get_activation_func(cfg_network['adjoint_net']['activation']))
                    modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

            self.adjoint_net = nn.Sequential(*modules).to(self.device)

        # Zero init for the last layer
        for module in reversed(self.adjoint_net):
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)
                nn.init.zeros_(module.bias)
                break

    def forward(self, state, param=None):
        if self.parametric:
            state_encoding = self.state_net(state.to(self.device))
            param_encoding = self.param_net(param.to(self.device))
            state_param_encodings = torch.cat([state_encoding, param_encoding], dim=-1)
            adjoint = self.adjoint_net(state_param_encodings)
        else:
            adjoint = self.adjoint_net(state.to(self.device))

        return adjoint

    def save(self, directory, episode_num=0, filename="adjoint_net"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        full_filename = filename + '_ep_' + str(episode_num)
        full_path = os.path.join(directory, "{}.pt".format(full_filename))
        torch.save(self.state_dict(), full_path)
        print(f"Adjoint net saved at {full_path}")

    def load(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint)
        self.to(device)
        print(f"Adjoint net loaded from {path}")
