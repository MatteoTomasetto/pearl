import torch
from fenics import *
from fenics_adjoint import *

class PDE(torch.autograd.Function):
    ''' Custom PyTorch autograd Function that encapsulates a single PDE solve and its adjoint for gradient computation. '''
    
    @staticmethod
    def forward(ctx, y_old, u, env):
        
        # Initialize tape
        tape = Tape()
        set_working_tape(tape)

        # Convert torch tensors to fenics functions
        y_old_fun = env.vec2fun(y_old, env.Y)
        u_fun = env.vec2fun(u, env.U)
        
        # Physics solve
        y_new_fun = env.compute_state(y_old_fun, u_fun) 

        # Store for backward
        ctx.env = env
        ctx.tape = tape
        ctx.y_old_fun = y_old_fun
        ctx.y_new_fun = y_new_fun
        ctx.u_fun = u_fun
        
        return env.fun2vec(y_new_fun)
    
    @staticmethod
    def backward(ctx, grad_output):
        
        # Initialize tape
        env = ctx.env
        set_working_tape(ctx.tape)

        # Convert torch tensors to fenics functions
        adj_value = env.vec2fun(grad_output, env.Y)

        # Compute VJPs
        dF_dy = compute_gradient(ctx.y_new_fun, Control(ctx.y_old_fun), adj_value = adj_value.vector())
        dF_du = compute_gradient(ctx.y_new_fun, Control(ctx.u_fun), adj_value = adj_value.vector())

        return (env.fun2vec(dF_dy).unsqueeze(0), env.fun2vec(dF_du).unsqueeze(0), None)

class Cost(torch.autograd.Function):
    ''' Custom PyTorch autograd Function that encapsulates the cost functional and its adjoint for gradient computation. '''

    @staticmethod
    def forward(ctx, y_new, u, env):
        
        # Initialize tape
        tape = Tape()
        set_working_tape(tape)
        
        # Convert torch tensors to fenics functions
        y_new_fun = env.vec2fun(y_new, env.Y)
        u_fun = env.vec2fun(u, env.U)
        
        # Compute cost
        J = assemble(env.compute_cost(y_new_fun, u_fun))
        
        # Store for backward
        ctx.env = env
        ctx.tape = tape
        ctx.y_new_fun = y_new_fun
        ctx.u_fun = u_fun
        ctx.J = J

        return torch.tensor(J, requires_grad=True).float()

    @staticmethod
    def backward(ctx, grad_output):
        
        # Initialize tape
        env = ctx.env
        set_working_tape(ctx.tape)
        
        # Compute VJPs
        dJ_dy = compute_gradient(ctx.J, Control(ctx.y_new_fun))
        dJ_du = compute_gradient(ctx.J, Control(ctx.u_fun))
        
        return (env.fun2vec(dJ_dy).unsqueeze(0) * grad_output, env.fun2vec(dJ_du).unsqueeze(0) * grad_output, None)