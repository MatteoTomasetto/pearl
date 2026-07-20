import os
import math
import numpy as np
import torch
import gymnasium as gym
from fenics import *
from fenics_adjoint import *
from utils.torch_fenics import PDE, Cost
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import imageio
from IPython.display import clear_output as clc
from IPython.display import display

set_log_level(LogLevel.ERROR)

class LeaderFollower_SingleAgent(gym.Env):
    def __init__(self, config): 

        self.device = config["device"]
        self.dtype = torch.float32

        # unpack config
        self.random_init = config["random_init"]
        self.parametric_target = config["parametric_target"]
        self.parametric_flow = config["parametric_flow"]
        self.follow_target = config["follow_target"]
        self.sparse_reward = config["sparse_reward"]
        self.dt = torch.tensor(config["dt"], device = self.device, dtype = self.dtype)
        self.T = torch.tensor(config["T"], device = self.device, dtype = self.dtype)
        self.soft_reward_scale = torch.tensor(config["soft_reward_scale"], device = self.device, dtype = self.dtype)
        self.soft_reward_damping = torch.tensor(config["soft_reward_damping"], device = self.device, dtype = self.dtype)
        self.save_plots = config["save_plots"]
        self.save_gifs = config["save_gifs"]   
        
        # define problem parameters
        self.Nt = math.ceil((self.T / self.dt).item())
        self.Ny = 4
        self.Nu = 2
        self.Np = 4 if self.parametric_flow else 2
        self.doublegyre_intensity = torch.tensor(0.1, device = self.device, dtype = self.dtype)
        self.doublegyre_amplitude = torch.tensor(0.25, device = self.device, dtype = self.dtype)
        self.doublegyre_frequency = torch.tensor(np.pi, device = self.device, dtype = self.dtype)
        self.goal_radius_sq = torch.tensor(0.001, device = self.device, dtype = self.dtype)
        self.control_scale = 0.2

        # define state
        self.state = None

        # define success         
        self.success = False

        # lists for states and controls
        self.states_plot = []
        self.controls_plot = []
        self.target_trajectory = []      

    def step(self, u):
        ''' Step function for the environment evolution '''
        # update the target position
        if self.follow_target:
            self.move_target()

        # set agent position
        agent_pos = torch.stack([self.state[0], self.state[1]])

        # advance agent position
        u = u[0] if u.dim() == 2 else u
        agent_pos = agent_pos + self.dt * self.compute_velocity(self.t, agent_pos, u)

        v = self.compute_velocity(self.t, agent_pos)
        
        # update state
        self.state = torch.stack([agent_pos[0], agent_pos[1], v[0], v[1]])

        # update time
        self.t += self.dt

        # compute cost and reward
        state_term = (self.state[0] - self.goal[0])**2 + (self.state[1] - self.goal[1])**2
        if self.sparse_reward:
            state_term = - self.soft_reward_scale * torch.exp(-self.soft_reward_damping * state_term)
        action_term = self.control_scale * (u[0]**2 + u[1]**2)
        if self.follow_target:
            soft_bonus = 0
        else:
            soft_bonus = self.soft_reward_scale * torch.exp(-self.soft_reward_damping * state_term)

        cost = state_term + action_term - soft_bonus
        reward = -cost

        # stores function for plotting
        self.states_plot.append(self.state.detach().cpu().numpy())
        self.controls_plot.append(u.detach().cpu().numpy())

        # set done and success 
        done = False
        if not self.follow_target:
            if state_term < self.goal_radius_sq:
                print(f"Target reached! -- Soft bonus {soft_bonus}")
                done = True
                self.success = True
        
        time_over = False
        if self.t >= self.T:
            done = True
            time_over = True

        info = {
            'state_cost': state_term.detach(),
            'action_cost': action_term.detach(),
            'reward': reward.detach(),
            'cost': cost.detach(),
            'success': self.success,
            'time_over': time_over
        }

        return self.state, reward, done, info
  
    def reset(self):
        ''' Reset environment to initial state and return initial observation ''' 
        # reset time and success
        self.t = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.success = False

        # reset state
        if self.random_init:
            agent_pos = torch.tensor([np.random.uniform(low = 0.1, high = 1.9),
                                       np.random.uniform(low = 0.1, high = 0.9)], device = self.device, dtype = self.dtype)
        else:
            agent_pos = torch.tensor([0.5, 0.5], device = self.device, dtype = self.dtype) # Default initial position

        v = self.compute_velocity(self.t, agent_pos)
        self.state = torch.tensor([agent_pos[0], agent_pos[1], v[0], v[1]], device = self.device, dtype = self.dtype)
        self.state = self.state.requires_grad_(True)

        # reset goal
        if self.parametric_target:
            self.goal = torch.tensor([np.random.uniform(low = 0.1, high = 1.9),
                                      np.random.uniform(low = 0.1, high = 0.9)], device = self.device, dtype = self.dtype)
        else:
            self.goal = torch.tensor([1.5, 0.5], device = self.device, dtype = self.dtype) # Default goal position

        # reset flow
        if self.parametric_flow:
            self.doublegyre_intensity = torch.tensor(np.random.uniform(low = 0.1, high = 0.4), device = self.device, dtype = self.dtype)
            self.doublegyre_frequency = torch.tensor(np.random.uniform(low = 0.2*np.pi, high = np.pi), device = self.device, dtype = self.dtype)
        else:
            self.doublegyre_intensity = torch.tensor(0.1, device = self.device, dtype = self.dtype) # Default intensity
            self.doublegyre_frequency = torch.tensor(np.pi, device = self.device, dtype = self.dtype) # Default frequency

        # store functions for plotting
        self.states_plot = [self.state.detach().cpu().numpy()]
        self.controls_plot = []
        if self.follow_target:
            self.target_trajectory = [self.goal.detach().cpu().numpy()]

        return self.state
    
    def move_target(self):
        ''' Move target in the double gyre flow '''
        v = self.compute_velocity(self.t, self.goal)
        self.goal = torch.stack([self.goal[0] + v[0] * self.dt, self.goal[1] + v[1] * self.dt])
        self.target_trajectory.append(self.goal.detach().cpu().numpy())
        return

    def initialize_trajectory(self):
        ''' Start short horizon with the current detached state '''
        if self.state is None:
            raise RuntimeError("Please reset the environment before.")
        
        self.state = self.state.detach().requires_grad_(True)

        return self.state

    
    def get_parameters(self):
        ''' Return environment parameters '''
        if self.parametric_flow:
            return torch.cat([self.goal, torch.tensor(self.doublegyre_intensity).unsqueeze(0), torch.tensor(self.doublegyre_frequency).unsqueeze(0)], dim = 0)
        else:
            return self.goal


    def compute_velocity(self, t, pos, u = torch.tensor([0.0, 0.0])):
        x = pos[..., 0]
        y = pos[..., 1]
        u1 = u[..., 0]
        u2 = u[..., 1]

        f = self.doublegyre_amplitude * torch.sin(self.doublegyre_frequency * t) * x ** 2 + (1 - 2 * self.doublegyre_amplitude * torch.sin(self.doublegyre_frequency * t)) * x
        df = 2 * self.doublegyre_amplitude * torch.sin(self.doublegyre_frequency * t) * x + 1 - 2 * self.doublegyre_amplitude * torch.sin(self.doublegyre_frequency * t)

        v1 = -torch.pi * self.doublegyre_intensity * torch.sin(torch.pi * f) * torch.cos(torch.pi * y) + self.control_scale * u1
        v2 =  torch.pi * self.doublegyre_intensity * torch.cos(torch.pi * f) * torch.sin(torch.pi * y) * df + self.control_scale * u2

        return torch.stack([v1, v2], dim = -1)
             
    
    def double_gyre_flow(self, x, y, t):
        ''' Double gyre flow over space and time'''

        xgrid, ygrid = np.meshgrid(x, y)  # spatial grid

        v1 = np.zeros((len(t), len(x), len(y)))  # horizontal velocity
        v2 = np.zeros((len(t), len(x), len(y)))  # vertical velocity

        amplitude = self.doublegyre_amplitude
        if amplitude.is_cuda:
            amplitude = amplitude.cpu()
        intensity = self.doublegyre_intensity
        if intensity.is_cuda:
            intensity = intensity.cpu()
        frequency = self.doublegyre_frequency
        if frequency.is_cuda:
            frequency = frequency.cpu()

        f = lambda x, t: amplitude * np.sin(frequency * t) * x ** 2 + x - 2 * amplitude * np.sin(frequency * t) * x

        # compute solution
        for i in range(len(t)):
            v1[i] = (-np.pi * intensity * np.sin(np.pi * f(xgrid, t[i])) * np.cos(np.pi * ygrid)).T
            v2[i] = (np.pi * intensity * np.cos(np.pi * f(xgrid, t[i])) * np.sin(np.pi * ygrid) * (2 * amplitude * np.sin(frequency * t[i]) * xgrid + 1.0 - 2 * amplitude * np.sin(frequency * t[i]))).T

        return v1, v2
    
    def vorticity(self, v1, v2, dx, dy):
        ''' Compute vorticity '''
        dv1_dy = np.gradient(v1, dy, axis = 1)
        dv2_dx = np.gradient(v2, dx, axis = 0)
        return dv2_dx - dv1_dy

    def render(self, save_dir, info = ""):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # spatial discretization
        nx = 50
        ny = 25
        Lx = 2.0
        Ly = 1.0
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        dx = Lx / nx
        dy = Ly / ny

        # casting
        dt = self.dt
        if dt.is_cuda:
            dt = dt.cpu()
        T = self.T
        if T.is_cuda:
            T = T.cpu()
        t = np.arange(0, T + dt, dt)

        if self.follow_target:
            goal = np.array(self.target_trajectory)
        else:   
            goal = self.goal
            if goal.is_cuda:
                goal = goal.cpu()
        
        states = np.array(self.states_plot)
        controls = np.array(self.controls_plot)

        # compute double gyre flow over space and time
        v1, v2 = self.double_gyre_flow(x, y, t)

        # flow rendering
        offset = 0.1
        if self.save_plots:
            fig, ax = plt.subplots()
            ax.contourf(x, y, self.vorticity(v1[-1], v2[-1], dx, dy).T, cmap = 'bone', levels = 100)
            ax.streamplot(x, y, v1[-1].T, v2[-1].T, color = 'black', linewidth = 1, density = 1)
            plt.axis('off')
            plt.axis([0 - offset, Lx + offset, 0 - offset, Ly + offset])
            plt.grid(True)

            # goal rendering
            if self.follow_target:
                ax.scatter(goal[0][0], goal[0][1], c = 'yellow', marker = "s", s = 120, alpha = 1.0, edgecolors = 'black', linewidths = 3, zorder = 10)
                ax.plot(goal[:, 0], goal[:, 1], color='yellow', linewidth=2, zorder=3, solid_capstyle='round', solid_joinstyle='round')
            else:
                ax.scatter(goal[0], goal[1], c = 'yellow', marker = "s", s = 120, alpha = 1.0, edgecolors = 'black', linewidths = 3, zorder = 4)

            # agent rendering
            ax.scatter(states[0][0], states[0][1], c = 'magenta', marker = "o", s = 120, edgecolors = 'black', linewidths = 3, zorder = 4)       
            ax.plot(states[:, 0], states[:, 1], color='magenta', linewidth=2, zorder=3, solid_capstyle='round', solid_joinstyle='round')

            plt.savefig(save_dir + '/' + 'navigation_' + info + ".png", dpi=300)
            plt.close(fig)

            # goal and agent rendering over time
            if self.follow_target:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.set_facecolor("#eaeaf1")

                ax.scatter(0, goal[0][0], c = 'darkkhaki', marker = "s", s = 120, alpha = 1.0, edgecolors = 'black', linewidths = 3, zorder = 10, label = "Target")
                ax.plot(np.linspace(0, self.T + self.dt, goal.shape[0]), goal[:, 0], color='darkkhaki', linewidth=3, zorder=3, solid_capstyle='round', solid_joinstyle='round')
                
                ax.scatter(0, goal[0][1], c = 'darkkhaki', marker = "s", s = 120, alpha = 1.0, edgecolors = 'black', linewidths = 3, zorder = 10)
                ax.plot(np.linspace(0, self.T + self.dt, goal.shape[0]), goal[:, 1], color='darkkhaki', linewidth=3, zorder=3, solid_capstyle='round', solid_joinstyle='round')

                ax.scatter(0, states[0][0], c = 'magenta', marker = "o", s = 120, edgecolors = 'black', linewidths = 3, zorder = 4, label = "Agent")       
                ax.plot(np.linspace(0, self.T + self.dt, states.shape[0]), states[:, 0], color='magenta', linewidth=3, zorder=3, solid_capstyle='round', solid_joinstyle='round')

                ax.scatter(0, states[0][1], c = 'magenta', marker = "o", s = 120, edgecolors = 'black', linewidths = 3, zorder = 4)       
                ax.plot(np.linspace(0, self.T + self.dt, states.shape[0]), states[:, 1], color='magenta', linewidth=3, zorder=3, solid_capstyle='round', solid_joinstyle='round')

                plt.ylim(0, 2)
                plt.grid(True, color='white', linewidth=2.0)
                for spine in ax.spines.values():
                    spine.set_color("white")
                ax.tick_params(axis="both", which="major", labelsize=15)
                ax.set_xticks([0, 20, 40, 60, 80, 100])
                ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
                ax.tick_params(axis="both", which="both", length=0)
                ax.set_xlabel('Time', labelpad=-5,  fontsize=17)
                leg = plt.legend(framealpha=1, facecolor='white', edgecolor='black', fontsize=15, handlelength=1.5)
                leg.get_frame().set_linewidth(2)
                plt.tight_layout()
                
                plt.savefig(save_dir + '/' + 'navigation_track_' + info + ".png", dpi=300)
                plt.close(fig)

        # create and save frames 
        if self.save_gifs:
            frames = []
            for i in range(0, states[:-1].shape[0], 5):
                # flow rendering
                fig, ax = plt.subplots()
                canvas = FigureCanvas(fig)
                ax.contourf(x, y, self.vorticity(v1[i], v2[i], dx, dy).T, cmap = 'bone', levels = 100)
                ax.streamplot(x, y, v1[i].T, v2[i].T, color = 'black', linewidth = 1, density = 1)
                plt.axis('off')
                plt.axis([0 - offset, Lx + offset, 0 - offset, Ly + offset])
                plt.grid(True)

                # goal rendering
                if self.follow_target:
                    ax.scatter(goal[i][0], goal[i][1], c = 'yellow', marker = "s", s = 120, alpha = 1.0, edgecolors = 'black', linewidths = 3, zorder = 4)
                    ax.plot(goal[:i, 0], goal[:i, 1], color='yellow', linewidth=2, zorder=3, solid_capstyle='round', solid_joinstyle='round')
                else:
                    ax.scatter(goal[0], goal[1], c = 'yellow', marker = "s", s = 120, alpha = 1.0, edgecolors = 'black', linewidths = 3, zorder = 4)
                
                # agent rendering
                ax.scatter(states[i][0], states[i][1], c = 'magenta', marker = "o", s = 120, edgecolors = 'black', linewidths = 3, zorder = 4)          
                ax.plot(states[:i, 0], states[:i, 1], color='magenta', linewidth=2, zorder=3, solid_capstyle='round', solid_joinstyle='round')

                # control rendering
                norm = np.linalg.norm(controls, axis = 1)
                plt.quiver(states[i, 0], states[i, 1], controls[i, 0], controls[i, 1], norm[i], scale = 8, cmap = plt.get_cmap('spring'), zorder = 2)         
                
                canvas.draw()
                frames.append(np.array(canvas.buffer_rgba()))

                plt.close(fig)

            imageio.mimsave(save_dir + '/' + 'navigation_' + info  + ".gif", frames, loop = 0, fps = 5)


class LeaderFollower_MeanField(gym.Env):
    def __init__(self, config):

        self.device = "cpu"
        
        # unpack config
        self.random_init = config["random_init"] 
        self.random_target = config["random_target"] 
        self.dt = config["dt"]
        self.T = config["T"]
        self.beta = config["beta"]
        self.beta_g = config["beta_g"] 
        self.save_plots = config["save_plots"]
        self.save_gifs = config["save_gifs"]

        # define problem parameters
        self.Nt = round(self.T / self.dt)
        self.Np = 2
        self.diffusion = 0.001
        self.doublegyre_intensity = 0.1
        self.doublegyre_amplitude = 0.25
        self.doublegyre_frequency = np.pi

        # define mesh
        self.mesh_size = 16
        self.mesh = RectangleMesh(Point(0, 0), Point(2, 1), 2*self.mesh_size, self.mesh_size) 
        self.mesh = refine(self.mesh)
        self.mesh_plot = RectangleMesh(Point(0, 0), Point(2, 1), self.mesh_size, self.mesh_size//2) 

        # define measures
        self.dx = Measure("dx", domain=self.mesh)
        self.ds = Measure("ds", domain=self.mesh)

        # define control space
        self.U = FunctionSpace(self.mesh, VectorElement('CG', self.mesh.ufl_cell(), 1))
        self.Nu = self.U.dim() 
        self.U_plot = FunctionSpace(self.mesh_plot, VectorElement('CG', self.mesh_plot.ufl_cell(), 1))
        self.U_coords = self.U.tabulate_dof_coordinates().reshape((-1, self.mesh.geometry().dim()))
        self.x1 = self.U_coords[::2, 0]
        self.x2 = self.U_coords[::2, 1]

        # define state space and state dofs
        self.Y = self.U.sub(0).collapse()
        self.Ny = self.Y.dim() 
        self.y = None

        # lists for states, controls and target trajectories
        self.states_plot = []
        self.controls_plot = []
        self.target_trajectory = []          

        # define cmap for control visualization
        self.white = colors.LinearSegmentedColormap.from_list("", ["white", "white"])
        ice = sns.color_palette("icefire", as_cmap=True).colors
        col = [ice[i] for i in np.concatenate((np.arange(128, 0, -20), np.arange(254, 160, -12)))]
        col.insert(0, "black")
        self.cmap = colors.LinearSegmentedColormap.from_list("", col)

    def step(self, u):
        ''' Step function for the environment evolution '''
        if self.done:
            raise RuntimeError("Environment is done. Please reset.")

        # update the target position
        self.move_target()

        # advance dynamics
        y_new = PDE.apply(self.y, u, self).to(self.device)
        
        # compute cost
        cost = Cost.apply(y_new, u, self).to(self.device)

        # stores function for plotting
        self.states_plot.append(self.vec2fun(y_new.detach(), self.Y))
        self.controls_plot.append(self.vec2fun(u.detach(), self.U))
          
        # update time
        self.t += float(self.dt)
        self.done = self.t >= self.T
    
        # compute reward for logging
        reward = -cost.item()     
    
        info = {"reward": reward}

        # update state
        self.y = y_new

        return y_new, -cost, self.done, info

    def compute_state(self, y_old, u):
        ''' Solve the Fokker-Planck equation with a double gyre transport term '''
        # compute transport term
        v = Function(self.U)
        v1, v2 = self.double_gyre_flow(self.t, self.x1, self.x2)
        v.vector()[:] = np.ravel(np.column_stack((v1.reshape(-1, 1), v2.reshape(-1, 1))))

        # compute control and velocity components
        u1, u2 = split(project(u, self.U))
        v1, v2 = split(project(v, self.U))
        
        # define test and trial functions
        w = TestFunction(self.Y)
        y = TrialFunction(self.Y)

        # define variational problems
        a = (inner(y, w) * self.dx 
            + 0.5 * self.dt * self.diffusion * inner(grad(y), grad(w)) * self.dx 
            - 0.5 * self.dt * y * (u1 + v1) * w.dx(0) * self.dx 
            - 0.5 * self.dt * y * (u2 + v2) * w.dx(1) * self.dx)

        L = (inner(y_old, w) * self.dx 
            - 0.5 * self.dt * self.diffusion * inner(grad(y_old), grad(w)) * self.dx 
            + 0.5 * self.dt * y_old * (u1 + v1) * w.dx(0) * self.dx 
            + 0.5 * self.dt * y_old * (u2 + v2) * w.dx(1) * self.dx)

        # solve the variational problem
        y_new = Function(self.Y)
        solve(a == L, y_new)
        
        return y_new

    def compute_cost(self, y, u):
        ''' Compute cost functional '''
        target_term = 0.5 * inner(y - self.target, y - self.target) * self.dx
        boundary_term = 0.5 * inner(y, y) * self.ds
        control_term = 0.5 * self.beta * (inner(u, u) * self.dx + inner(grad(u), grad(u)) * self.dx)

        return target_term + boundary_term + control_term

    def reset(self):
        ''' Reset environment to initial state and return initial observation ''' 
        # reset time and cost
        self.t = 0.0
        self.done = False

        # reset state
        if self.random_init:
            self.y0_pos = torch.tensor([self.random_uniform(0.3, 1.7), self.random_uniform(0.3, 0.7)], dtype=torch.float32, device=self.device)
        else:
            self.y0_pos = torch.tensor([0.5, 0.5], dtype=torch.float32, device=self.device) # Default initial position
        y_fun = self.gaussian_generator(self.y0_pos)
        self.y = self.fun2vec(y_fun).requires_grad_(True)
        
        # reset target
        if self.random_target:
            self.target_pos = torch.tensor([self.random_uniform(0.1, 1.9), self.random_uniform(0.1, 0.9)], dtype=torch.float32, device=self.device)
        else:
            self.target_pos = torch.tensor([1.5, 0.5], dtype=torch.float32, device=self.device) # Default target position
        self.target = self.gaussian_generator(self.target_pos) 

        # store functions for plotting
        self.states_plot = [y_fun]
        self.controls_plot = []
        self.target_trajectory = [self.target_pos.clone()]

        return self.y

    def initialize_trajectory(self):
        ''' Start short horizon with the current detached state '''
        if self.y is None:
            raise RuntimeError("Please reset the environment before.")
        self.y = self.y.detach().requires_grad_(True)
        return self.y
    
    def get_parameters(self):
        ''' Return environment parameters '''
        return self.target_pos

    def random_uniform(self, a, b):
        ''' Generate a random number uniformly between a and b '''
        return (b - a) * torch.rand(1) + a

    def gaussian_generator(self, center):
        '''0 Generate Gaussian distribution with mean equal to center '''
        x1 = center[0]
        x2 = center[1]
        y = Expression('10 / pi * exp(- 10*(x[0] - x0)*(x[0] - x0) - 10*(x[1] - x1)*(x[1] - x1))', degree = 1, x0 = Constant(x1), x1 = Constant(x2))
        y = interpolate(y, self.Y)
        return y

    def double_gyre_flow(self, t, x1, x2):
        ''' Evaluate the double gyre velocity field at given time t and spatial coordinates x1, x2 '''
        f = lambda x, t: self.doublegyre_amplitude * np.sin(self.doublegyre_frequency * t) * x**2 + x - 2 * self.doublegyre_amplitude * np.sin(self.doublegyre_frequency * t) * x
        v1 = (-np.pi * self.doublegyre_intensity * np.sin(np.pi * f(x1, t)) * np.cos(np.pi * x2)).T
        v2 = (np.pi * self.doublegyre_intensity * np.cos(np.pi * f(x1, t)) * np.sin(np.pi * x2) * (2 * self.doublegyre_amplitude * np.sin(self.doublegyre_frequency * t) * x2 + 1.0 - 2 * self.doublegyre_amplitude * np.sin(self.doublegyre_frequency * t))).T
        return v1, v2

    def move_target(self):
        ''' Move target in the double gyre flow '''
        v1, v2 = self.double_gyre_flow(self.t, self.target_pos[0], self.target_pos[1])
        self.target_pos = torch.stack([self.target_pos[0] + v1 * self.dt, self.target_pos[1] + v2 * self.dt])
        self.target = self.gaussian_generator(self.target_pos) 
        self.target_trajectory.append(self.target_pos.clone())
        return
    
    def fun2vec(self, fun):
        ''' Convert fenics function into a torch tensor '''
        return torch.from_numpy(fun.vector()[:]).float().to(self.device)

    def vec2fun(self, vec, fun_space):
        ''' Convert torch tensor to fenics function '''
        fun = Function(fun_space)
        fun.vector()[:] = vec.cpu().numpy().flatten()
        return fun

    def render(self, directory, info = ""):
        ''' Render the state and control trajectories '''
        # define paths
        if not os.path.exists(directory):
            os.makedirs(directory)
        state_filename = "state_" + str(info)
        state_path = os.path.join(directory, state_filename)
        control_filename = "control_" + str(info)
        control_path = os.path.join(directory, control_filename)

        target_trajectory = torch.stack(self.target_trajectory)

        # create and save frames
        state_frames = []
        control_frames = []
        for i in range(len(self.states_plot)):
            # state rendering
            y = self.states_plot[i]

            fig = plt.figure()
            canvas = FigureCanvas(fig)
            plot(y, self.Y, cmap = "jet")

            plt.scatter(target_trajectory[i,0],
                       target_trajectory[i,1],
                       color='yellow',
                       marker='s',
                       s=100,
                       edgecolor='black',
                       linewidth=3,
                       zorder=5)
            plt.plot(target_trajectory[:i+1,0], target_trajectory[:i+1,1], color='yellow', linewidth=2, zorder=3, solid_capstyle='round', solid_joinstyle='round')

            plt.axis('off')
            if self.save_plots:
                plt.savefig(state_path + "_frame_" + str(i) + ".png")
            if self.save_gifs:
                canvas.draw()
                state_frames.append(np.array(canvas.buffer_rgba()))
                plt.close(fig)
            if not self.save_plots and not self.save_gifs:
                display(fig)
                plt.close(fig) 
                clc(wait=True)
            
            # control rendering
            if i < len(self.controls_plot):
                u = self.controls_plot[i]
                u.set_allow_extrapolation(True)
                fig = plt.figure()
                canvas = FigureCanvas(fig)
                plot(sqrt(u**2), cmap = self.cmap)
                plot(fenics.project(u, self.U_plot), cmap = self.white, alpha = 0.8)
                plt.scatter(target_trajectory[i,0],
                           target_trajectory[i,1],
                           color='yellow',
                           marker='s',
                           s=100,
                           edgecolor='black',
                           linewidth=3,
                           zorder=5)
                plt.plot(target_trajectory[:i+1,0], target_trajectory[:i+1,1], color='yellow', linewidth=2, zorder=3, solid_capstyle='round', solid_joinstyle='round')

                plt.axis('off')
                if self.save_plots:
                    plt.savefig(control_path + "_frame_" + str(i) + ".png")
                if self.save_gifs:    
                    canvas.draw()
                    control_frames.append(np.array(canvas.buffer_rgba()))
                plt.close(fig)
                clc(wait=True)

        # save gifs
        if self.save_gifs:
            imageio.mimsave(state_path + '.gif', state_frames, loop = 0, fps = 5)
            imageio.mimsave(control_path + '.gif', control_frames, loop = 0, fps = 5)

