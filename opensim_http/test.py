from osim.env.run import RunEnv
import numpy as np
from pykalman import UnscentedKalmanFilter
import matplotlib.pyplot as plt

def transition_function(state, noise):
    H = np.array([[1, 1],[0,1]])
    return np.dot(H, state) + noise

def observation_function(state, noise):
    C = np.eye(2)
    return np.dot(C, state) + noise


transition_covariance = np.eye(2)
random_state = np.random.RandomState(0)
observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.01
initial_state_mean = [0, 0]
initial_state_covariance = [[0.1, 0.01], [0.01, 0.1]]

kf = UnscentedKalmanFilter(
    transition_function, observation_function,
    transition_covariance, observation_covariance,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)




env = RunEnv(visualize=False)

observation = env.reset(difficulty = 2)

ball_log = []

pelvis_x = observation[1]
ball_relative = observation[38]
ball_absolute = pelvis_x + ball_relative

ball_log.append(ball_absolute)
filtered_state_mean = np.array([observation[6],observation[12]])
filtered_state_covariance = np.array(initial_state_covariance)
a = []
b = []
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    measurement = np.array([observation[6],observation[12]])
    filtered_state_mean, filtered_state_covariance = kf.filter_update(filtered_state_mean, filtered_state_covariance, measurement)
    
    #print(measurement)
    #print(filtered_state_mean)
    a.append(measurement[0])
    b.append(filtered_state_mean[0])
    pelvis_x = observation[1]
    ball_relative = observation[38]
    ball_absolute = pelvis_x + ball_relative

    ball_log.append(ball_absolute)

    if done:
        env.reset()
        break

plt.plot(a, label="observation");plt.plot(b,label="update");
plt.legend(shadow=True, fancybox=True)
plt.show()
