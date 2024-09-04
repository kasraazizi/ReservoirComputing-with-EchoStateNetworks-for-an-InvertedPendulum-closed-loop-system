class Controller():
    def __init__(self):
        """
        Attributes: 
        P (float): Proportional gain. 
        I (float): Integral gain. 
        D (float): Derivative gain. 
        e (float): Current error. 
        dt (float): Time difference between control cycles. 
        
        prev_error (float): Error from the previous control cycle. 
        integral_error (float): Cumulative error over time. 
        
        angle_setpoint (float): Desired angle setpoint. 
        pos_setpoint (float): Desired position setpoint. 
        
        a (float): Weight for angle error in the total error calculation. 
        b (float): Weight for position error in the total error calculation.
        """
        self.P = 0
        self.I = 0
        self.D = 0
        self.e = 0
        self.prev_error = 0
        self.integral_error = 0
        self.dt = 1
        
        self.angle_setpoint = 0
        self.pos_setpoint = 0
                       
        self.a = 1
        self.b = 1
        
    
    def PID(self, measure, prev_error, integral_error):
        """
        This function calculates the control output (u) based on the Proportional-Integral-Derivative (PID) control algorithm.
          
        Parameters:
        measure (list): A list containing the current angle and position measurements. The first element is the angle, and the second element is the position.
        prev_error (float): The error from the previous control cycle.
        integral_error (float): The cumulative error over time.

        Returns:
        tuple: A tuple containing the control output (u), the total error, and the derivative error. The total error is the sum of the angle and position errors, and the derivative error is the rate of change of the total error.
        """

        
        total_error = self.a*(self.angle_setpoint - measure[0]) + self.b*(self.pos_setpoint - measure[1])    
        
        integral_error = integral_error + total_error * self.dt
        derivative = (total_error  - prev_error)/self.dt
                    
        u = self.P*total_error  + self.I* self.integral_error + self.D*derivative
        
        return u , total_error, integral_error
        
   
    