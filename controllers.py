class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3
  
'''
1. Implement simple PID controller 
  - Tune parameters, observe effects of lataccel_cost and jerk_cost 
  
2. Implement an MPC controller 

3. Implement a controller that uses RL 

4. Improve controller optimizing total_cost 
'''

class PIDController(BaseController):
  def __init__(self, kp=0.1, ki=0.01, kd=0.05):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.prev_error = 0
    self.integral = 0
    
  def update(self, target_lataccel, current_lataccel, state):
    error = target_lataccel - current_lataccel
    self.integral += error
    derivative = error - self.prev_error
    
    output = self.kp * error + self.ki * self.integral + self.kd * derivative
    self.prev_error = error
    return output 

class MPCController(BaseController): 
  def update(self, target_lataccel, current_lataccel, state):
    pass


CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController,
}
