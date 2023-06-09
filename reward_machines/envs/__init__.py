from gymnasium.envs.registration import register

register(
    id='Obstacle-v0',
    entry_point='reward_machines.envs.grids.grid_world:ObstacleRMEnv',
    max_episode_steps=1000
)