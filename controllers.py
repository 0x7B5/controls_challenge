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

import numpy as np
import cvxpy as cp
import onnxruntime as ort

class MPCController:
    def __init__(self, model_path="./models/tinyphysics.onnx", horizon=10, dt=0.1):
        # Load the ONNX model
        self.model = ort.InferenceSession(model_path)
        self.horizon = horizon
        self.dt = dt
        self.initial_state = cp.Parameter(3)  # Assuming state vector has 3 elements
        self.target = cp.Parameter(horizon)  # target lateral accelerations
        self.setup_mpc_problem()

    def setup_mpc_problem(self):
        # Define the optimization variables
        self.u = cp.Variable(self.horizon)  # control actions (steer commands)
        self.x = cp.Variable((self.horizon + 1, 3))  # states (lateral acceleration, velocity, etc.)

        # Define the cost function components
        self.target = cp.Parameter(self.horizon)  # target lateral accelerations
        cost = 0
        for t in range(self.horizon):
            cost += cp.sum_squares(self.x[t+1, 0] - self.target[t])  # Minimize error
            cost += 0.1 * cp.sum_squares(self.u[t])  # Control effort penalty

        # Define constraints
        constraints = [self.x[0, :] == self.initial_state]  # initial state constraint
        for t in range(self.horizon):
            constraints += [
                self.x[t+1] == self.x[t] + self.dt * self.model.predict(self.x[t], self.u[t]),  # Dynamics
                cp.abs(self.u[t]) <= 2  # Steering angle limits
            ]

        # Create optimization problem
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def update(self, initial_state, target_trajectory):
        self.initial_state.value = initial_state
        self.target.value = target_trajectory
        result = self.problem.solve()
        if result is None:
            raise Exception("MPC problem is infeasible")
        return self.u.value[0]  # Return the first control action



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
  'mpc': MPCController,
}
