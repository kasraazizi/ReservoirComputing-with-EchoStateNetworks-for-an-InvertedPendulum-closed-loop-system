import torch

class CartPole_measurement():
    def __init__(self):
        """
        Initialize the CartPole_measurement class.
        
        This class simulates a measurement system for a cart-pole system. It includes methods for simulating
        accelerometer, statemeter, and adding white noise to the measurements.
        
        Attributes:
        device (torch.device): The device on which the computations will be performed. Default is 'cpu'.
        mean (float): The mean value of the white noise. Default is 0.
        std (float): The standard deviation of the white noise. Default is 0.01.
        data_size (int): The size of the data. Default is 0.
        bias (torch.Tensor): The bias added to the measurements. Initialized as a tensor of size data_size.
        """
        self.device = torch.device('cpu')
        self.mean = 0
        self.std = 0.01
        self.data_size = 0
        self.bias = torch.tensor(self.data_size)

    def accelerometer(self, acceleration, velocity, position, dt):
        """
        This function takes the current acceleration, velocity, position, and time step (dt) as input,
        simulates an accelerometer measurement by adding white noise to the acceleration, updating the velocity and position,#+
        and adding a bias to the position. The white noise is generated using the `white_noise` method.

        Parameters:
        ----------
            
        acceleration : torch.Tensor
            The current acceleration of the cart-pole system. It should be a 1D tensor of size (1,).#+
        velocity : torch.Tensor
            The current velocity of the cart-pole system. It should be a 1D tensor of size (1,).#+
        position : torch.Tensor
            The current position of the cart-pole system. It should be a 1D tensor of size (1,).#+
        dt : float
            The time step for the simulation. It should be a positive scalar value.#+

        """
       
        acceleration = acceleration + self.white_noise(acceleration.size()).to(self.device)
        velocity = velocity + acceleration * dt
        position = position + velocity * dt +  self.bias 
        
        return torch.row_stack([position, velocity, acceleration])
       
    def statemeter(self, state):
        """
        Simulate a statemeter measurement for the cart-pole system.

        This function takes the true state of the cart-pole system as input, adds white noise to the state,
        and returns the simulated measurement. The white noise is generated using the `white_noise` method.

        Parameters:
        ----------
        state : torch.Tensor
            The true state of the cart-pole system. It should be a 1D tensor of size (n,) where n is the number of state variables.

        Returns:
        -------
        measured_state : torch.Tensor
            The simulated measurement of the cart-pole system state. It is a 1D tensor of size (n,) with white noise added.
        """
        measured_state = (state + self.white_noise(state.size())).to(self.device)        
        return measured_state

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
        
        noisy_data = (self.mean + torch.randn(size)*self.std**2 ).to(self.device)
        return noisy_data
