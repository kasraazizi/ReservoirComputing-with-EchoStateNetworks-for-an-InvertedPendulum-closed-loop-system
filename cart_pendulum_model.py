import torch
from warnings import warn
from torch import cos, sin


class CartPole_motion():
    '''Continuous version of the OpenAI Gym cartpole
    Inspired by: https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8'''
    def __init__(self):      
        self.device =  torch.device('cpu') 
        self.gravity = 9.81
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.model_std = 0
        self.model_mean = 0
        
    def dynamics(self, θ, dθ, u):
        """
        Calculate the continuous equation of motion of the cartpole.

        Parameters:
        ----------
        θ : torch.Tensor
            Current pole's angle.
        dθ : torch.Tensor
            Current pole's angular velocity.
        u : torch.Tensor
            Current control input.
            
        Returns:
        -------
        torch.Tensor
            A tensor containing the second derivatives of the pole's angle (ddθ) and the cart's velocity (ddx).
        """
        
        # Auxiliary variables
        cosθ, sinθ = torch.cos(θ), torch.sin(θ)
        temp = (u + self.polemass_length * dθ**2 * sinθ) / self.total_mass
        
        # Differential Equations
        ddθ = (self.gravity * sinθ - cosθ * temp) / \
                (self.length * (4.0/3.0 - self.masspole * cosθ**2 / self.total_mass))
        ddx = temp - self.polemass_length * ddθ * cosθ / self.total_mass

        return torch.row_stack([ddθ, ddx])


    def motion_model(self,state, velocity,  u, dt):
        """
        This function implements the continuous time dynamics of the cartpole system.

        Parameters
        ----------
        dt : float
            Time step size.
        velocity : torch.Tensor
            Previous Velocity of the cart.
        state : torch.Tensor
            Previous State of the cart.
        u : float
            Current Control input.

        Returns
        -------
        state: torch.Tensor
            Current state of the cartpole.        
        """
        accel = self.dynamics(state[0], velocity[0], u) + self.white_noise(state.size()).to(self.device)
        velocity = velocity + dt*accel
        state = state + dt*velocity
     
        return torch.row_stack([state, velocity, accel])

   
    def white_noise(self, size):
        """
        Generate a tensor of white noise with the given size.

        This function uses PyTorch's randn function to generate a tensor of random numbers
        from a standard normal distribution. The generated noise is then scaled by the
        model's standard deviation and shifted by the model's mean.

        Parameters:
        ----------
        size : tuple or torch.Size
            The size of the output tensor.

        Returns:
        -------
        noisy_data : torch.Tensor
            A tensor of white noise with the given size.
        """
        noisy_data = self.model_mean + torch.randn(size)*self.model_std**2        
        return noisy_data
