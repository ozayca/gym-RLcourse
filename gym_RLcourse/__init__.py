from gym.envs.registration import register

register(id='Corridor-v0',
         entry_point='gym_RLcourse.envs:CorridorEnv',
)

register(id='ShortCorridor-v0',
         entry_point='gym_RLcourse.envs:ShortCorridorEnv',
)

register(id='DynaMaze-v0',
         entry_point='gym_RLcourse.envs:DynaMazeEnv',
         kwargs={'maze_type' : 'DM'},
)
