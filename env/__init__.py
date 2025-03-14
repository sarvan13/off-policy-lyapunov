from gymnasium.envs.registration import register

register(
    id="Quadrotor-v1",
    entry_point="env.quad.quad_rotor:QuadRateEnv",  # Adjust as per your directory structure
)

register(
    id="Quadrotor-Still-v1",
    entry_point="env.quad.quad_rotor_still:QuadStillEnv",  # Adjust as per your directory structure
)