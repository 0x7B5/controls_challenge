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


#   def __init__(self, kp=0.10726307, ki=0.05933246, kd=0.00414424): (just trained on data file 0000)

# 0.13209257, 0.01741065



# kp=0.05580879, ki=0.06304656, kd=0 <- PSO trained on 1 file, 31.35
# kp=0.13115483, ki=0.06984474, kd=0.06984474 <- GA trained on 1 file, 37.12

# kp=0.05, ki=0.10485977, kd=0 <- PSO trained on random files, 30.43


# kp=0.05, ki=0.0779512, kd=0.04458376 <- best I think
class PIDController(BaseController):
  def __init__(self, kp=0.05, ki=0.0779512, kd=0.04458376):
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

      
'''
Max value of update is 2.0 because max revolutions for most cars is around 2. This is in radians? I think 
'''




'''

"PID is reactive control. It is lagging behind the setpoint. Upside being that it doesn't rely on a lot of data to learn.

RL is proactive control. It is optimizing for cumulative feedback into the future. This needs data and models to learn.

In my research (with UAV flight and temperature control) I've found that RL is better as a supervisory mechanism. The time and safety constraints in process control are sometimes too tight for a RL controller. However a RL agent that can modify the setpoint at a lower frequency for PID to track is a good bet."

https://old.reddit.com/r/reinforcementlearning/comments/113h9wi/is_rl_for_process_control_really_useful/
'''
CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PIDController,
  # 'mpc': MPCController,
}
