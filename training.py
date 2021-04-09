from setuptools import setup
from snake_env import Snake_Env
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

def human_playing():
	env = Snake_Env(server=False)
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
	env = Snake_Env(server = False)
	# env = make_vec_env(lambda: env, n_envs=4, monitor_dir="./vec")
	env = Monitor(env, "1e7_bw_dqn")
	obs = env.reset()
	model = DQN("CnnPolicy", env, verbose=1, optimize_memory_usage=True, buffer_size = 500000)
	model.learn(total_timesteps=1e7)
	model.save("1e7_bw_dqn")

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
	model = PPO.load("./positivereward", env=env)
	obs = env.reset()
	for i in range(1000):
		action, _state = model.predict(obs, deterministic=True)
		#action = env.action_space.sample()
		#print(action)
		obs, reward, done, info = env.step(action)
		env.render()
		if done:
			env.reset()

# human_playing()
# ai_playing()
ai_eval()