class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3
  
#   def __init__(self, kp=0.10726307, ki=0.05933246, kd=0.00414424): (just trained on data file 0000)
# kp=0.05580879, ki=0.06304656, kd=0 <- PSO trained on 1 file, 31.35
# kp=0.13115483, ki=0.06984474, kd=0.06984474 <- GA trained on 1 file, 37.12
# kp=0.05, ki=0.10485977, kd=0 <- PSO trained on random files, 30.43
# kp=0.05, ki=0.0779512, kd=0.04458376 <- best I think
class PIDController(BaseController):
  def __init__(self, kp=0.08010998, ki=0.14959143, kd=0.02125796):
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
