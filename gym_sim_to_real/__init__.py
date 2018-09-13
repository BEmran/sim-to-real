import logging
from gym.envs.registration import register
from gym import envs
logger = logging.getLogger(__name__)
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

if not 'Qube-v0' in env_ids:
    
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
    
    register(
        id = 'QubeMotorAngle-v0',
        entry_point = 'gym_sim_to_real.envs:QubeMotorAngleEnv',
        timestep_limit = 1000,
        reward_threshold = 1.0,
        nondeterministic = True,
    )


