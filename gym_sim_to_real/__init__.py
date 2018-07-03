import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id = 'Qube-v0',
    entry_point = 'gym_sim_to_real.envs:QubeEnv',
    timestep_limit = 1000,
    reward_threshold = 1.0,
    nondeterministic = True,
)

register(
    id = 'QubeMotor-v0',
    entry_point = 'gym_sim_to_real.envs:QubeMotorEnv',
    timestep_limit = 1000,
    reward_threshold = 1.0,
    nondeterministic = True,
)

