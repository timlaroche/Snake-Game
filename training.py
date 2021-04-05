from setuptools import setup
import snake_env
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor

def human_playing():
	env = snake_env.Snake_Env(server=False)
	env.reset()
	env.step(action=[])
	while env.flag:
		actions = env.get_actions()
		obs, reward, done, info = env.step(action=actions)
		print(f"""
			obs: {obs}
			reward: {reward}
			done: {done}
			info: {info}
			""")
		env.render()
		env.close()

def ai_playing():
	env = snake_env.Snake_Env(server = True)
	env = Monitor(env, "training")
	obs = env.reset()
	model = PPO("CnnPolicy", env, verbose=2, learning_rate=1e-4, device="cuda:0")
	model.learn(total_timesteps=1e6)
	model.save("cnn_model")

	# for i in range(1000):
	# 	# action, _state = model.predict(obs, deterministic=True)
	# 	action = env.action_space.sample()
	# 	# print(action)
	# 	obs, reward, done, info = env.step(action)
	# 	env.render()
	# 	if done:
	# 		env.reset()

human_playing()