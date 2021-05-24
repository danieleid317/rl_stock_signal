"""Project acts as a sentiment indicator for particular stocks. It can learn what it
looks like to be profitable in the future n timesteps.  Output is predicted direction
for next n timesteps."""

import yfinance as yf
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import random


from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


""" Define the functions that will be turning raw price data into useful features,
and for helping to score our output"""

def last_n_ppc(table):
  n = 60
  try:
    closes = table['Close'].ewm(span = 3 ,adjust = False).mean()
    opens = table['Open'].ewm(span = 3 ,adjust = False).mean()
    close = closes[-1:][0]
    open = opens[-60:-59][0]
    ppc =  ((close-open) / open+.00001)
    if math.isnan(ppc):
      ppc = 0.0
    return ppc
  except:
    return 0.0

def n_step_return(table):
  n_steps = 30
  try:
    closes = table['Close'].ewm(span = 3 ,adjust = False).mean()
    opens = table['Open'].ewm(span = 3 ,adjust = False).mean()
    close = closes[-1:][0]
    open = opens[-30:-29][0]
    ppc =  ((close-open) / open+.00001)
    if math.isnan(ppc):
      ppc = 0.0
    return ppc
  except:
    return 0.0

def cmf(table):
  try:
    i = 25
    mfv_21 = 0
    vol_21 = 0
    while i >= 1:
      mfm = ((table['Close'][-i:][0]  -  table['Low'][-i:][0]) - (table['High'][-i:][0] - table['Close'][-i:][0])) /  ((table['High'][-i:][0] - table['Low'][-i:][0]) +.0001)
      vol = table['Volume'][-i:][0]
      mfv_21 += (vol * mfm)
      vol_21 += vol
      i -= 1
    cmf = float(mfv_21 / (vol_21+.0001))
    return cmf
  except:
    return 0.0

def macd_hist(table):
  try:
    shortperiod = table['Open'].ewm(span = 12 ,adjust = False).mean()
    longperiod = table['Open'].ewm(span = 26 ,adjust = False).mean()
    macd = shortperiod - longperiod
    exp3 = macd.ewm(span=9, adjust=False).mean()
    return ((macd.iloc[-1:][0] - exp3.iloc[-1:][0]))
  except:
    return 0.0

# include current volume

df = pd.read_csv('highvol.txt')
tickerlist = []
for value in df.values:
  tickerlist.append(value[0])


"""Custom environment to simulate continuos flow of stock data per episode"""

class TrainingStockMarket(py_environment.PyEnvironment):
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(3,), dtype=np.float32, minimum=0, name='observation')
    self.indexer = -150
    self.total_change = 0.0
    self.table = self.init_table()
    self._episode_ended = False
    self.step_count = 0
    self._state = 0

  def _reset(self):
    self.indexer = -150
    self.total_change = 0.0
    self._episode_ended = False
    self.step_count = 0
    self._state = self.make_features(self.table[:self.indexer])
    return ts.restart(np.array(self._state, dtype=np.float32))

  def make_features(self , table):
    #feature creation is averagte of last two timesteps in contrast to frame stacking
    #or some other time of pooling function
    try:
      cm = (cmf(table) + cmf(table[:-1]))/2
      mcd = (macd_hist(table) + macd_hist(table[:-1]))/2
      ppc = (last_n_ppc(table) + last_n_ppc(table[:-1]))/2
      return [ cm, ppc ,mcd ]
    except:
      return [ 0.0 , 0.0, 0.0 ]

  def init_table(self):
    try:
      table = yf.Ticker(tickerlist[random.randint(0,len(tickerlist)-1)]).history(period='3mo',interval='1h',)
      table = table[:-1]
      return table
    except:
      table = yf.Ticker(tickerlist[random.randint(0,len(tickerlist)-1)]).history(period='3mo',interval='1h',)
      table = table[:-1]
      return table

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _step(self, action):
    if self._episode_ended == True:
      return self.reset()
    self.step_count += 1
    self.indexer += 1
    self._state = self.make_features(self.table[:self.indexer])

    # termination condition for maxsteps
    if self.step_count >= 120 :
      self._episode_ended = True
      return ts.termination(np.array(self._state, dtype=np.float32), reward = 0 )

    if action == 0:
      #long
      reward = n_step_return(self.table[:self.indexer+29])
      #self.total_change += reward
      return ts.transition(np.array(self._state, dtype=np.float32), reward = reward, discount=0.9997)

    elif action == 1:
      #short
      reward = - n_step_return(self.table[:self.indexer+29])
      #self.total_change += -reward
      return ts.transition(np.array(self._state, dtype=np.float32), reward = reward , discount=0.9997)



from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, q_rnn_network
import tf_agents
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
tf.compat.v1.enable_v2_behavior()

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network

train_py_env = TrainingStockMarket()
train_tf_env = tf_py_environment.TFPyEnvironment(train_py_env)

eval_py_env = TrainingStockMarket()
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_tf_env.observation_spec(),
    train_tf_env.action_spec(),
    fc_layer_params=(16,16,),
    )

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=.002)

train_step_counter = tf.compat.v2.Variable(0)

agent = reinforce_agent.ReinforceAgent(
    train_tf_env.time_step_spec(),
    train_tf_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_tf_env.time_step_spec(),
                                                train_tf_env.action_spec())

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=1,
    max_length=50000)

def collect_episodes(environment, policy, num_episodes = 20, calc_returns = False):
  episode_counter = 0
  environment.reset()
  epoch_return = 0
  while episode_counter < num_episodes:
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)
    epoch_return += next_time_step.reward.numpy()[0]
    if traj.is_boundary():
      episode_counter += 1
  if calc_returns:
    average_returns = (epoch_return/num_episodes)
    eval_history.append(average_returns)
    print('average eval return :', np.mean(eval_history[-12:]))

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)
num_epochs = 500
returns_history = []
eval_history = []
for _ in range(num_epochs):
  # Collect a few episodes using collect_policy and save to the replay buffer.
  train_py_env = TrainingStockMarket()
  train_tf_env = tf_py_environment.TFPyEnvironment(train_py_env)
  collect_episodes(train_tf_env, collect_policy, num_episodes = 1 )

  # Use data from the buffer and update the agent's network.
  experience = replay_buffer.gather_all()
  train_loss = agent.train(experience)
  replay_buffer.clear()

  step = agent.train_step_counter.numpy()

  if step % 4 == 0:
    eval_py_env = TrainingStockMarket()
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    print('step: ', step ,'epoch :', _)
    collect_episodes(eval_tf_env,eval_policy,num_episodes=1,calc_returns = True)


"""Evaluation loop is outputting sentiment."""
