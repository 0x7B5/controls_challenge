class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3
  
class PIDController(BaseController):
  def __init__(self, kp=0.045, ki=0.09501335, kd=-0.03649474):
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

      
CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController,
}
