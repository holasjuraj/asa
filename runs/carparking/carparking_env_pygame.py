from gym.utils.play import play

from garage.envs.box2d import CarParkingEnv

env = CarParkingEnv(
    random_start_range=0.25
)
play(env, zoom=3)
