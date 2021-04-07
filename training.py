from setuptools import setup
from snake_env import Snake_Env
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

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
	env = Snake_Env(server = True)
	env = make_vec_env(lambda: env, n_envs=4, monitor_dir="./vec")
	obs = env.reset()
	model = PPO("CnnPolicy", env, verbose=2, device="cuda:0")
	model.learn(total_timesteps=1e6)
	model.save("100reward")

	# for i in range(1000):
	# 	# action, _state = model.predict(obs, deterministic=True)
	# 	action = env.action_space.sample()
	# 	# print(action)
	# 	obs, reward, done, info = env.step(action)
	# 	env.render()
	# 	if done:
	# 		env.reset()


def ai_eval():
	env = Snake_Env(server = False)
	model = PPO.load("negativealive", env=env)
	obs = env.reset()
	for i in range(1000):
		action, _state = model.predict(obs, deterministic=True)
		#action = env.action_space.sample()
		#print(action)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			env.reset()

ai_playing()
# ai_eval()