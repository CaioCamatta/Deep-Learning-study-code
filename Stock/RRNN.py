import tensorflow as tf
import numpy as np
import gym
import matplotlib.pylab as plt
import pandas as pd
from collections import deque

## PARAMETERS
gamma = .99
stack_size = 20 # deque size

## MAKE DATA
data_points = 401
x = np.linspace(0, 10*np.pi, data_points)
df = pd.DataFrame({'date':range(data_points), 'price':np.power(np.sin(x)+2,1/1.6)})

## DEFINE MKD
possible_actions = [[0,1],[1,0]] # 0% in, 100% in

class Environment:
    """ To run an entire episode do:
            1. env.step() x 400
            2. env.discounted_rewards()
            3. env.reset() """
    def __init__(self, dataframe):
        self.current_step = stack_size          # initial step

        self.df = dataframe             # holds the dataframe
        self.df_len = len(dataframe)    # length of the dataframe

        # This deque (list with limited length) holds the previus N (stack_size) data points.
        # e.g. if the deque contains data from steps 200 -> 219 the next will do so for steps 201 -> 220.
        self.stack = deque(df.price.values[0:self.current_step], maxlen=stack_size)

        # Holds trading history, (step, price, action) tuples
        self.ep_memory = []


    def step(self, action):
        """ Return next state and whether the dataframe is finished or not (done) """
        done = False

        # Done = true if we ran out of data.
        if self.current_step == self.df_len-1:
            done = True

        # Append current price to stack
        current_price = self.df.price.values[self.current_step]
        self.stack.append(current_price)

        # Append data to episode memory
        self.ep_memory.append((self.current_step, current_price, action))

        # Define state
        state = np.array([self.stack])

        # Add to step
        self.current_step += 1

        return state, done


    def calculate_rewards(self):
        """ Returns list of discounted rewards.
            Must be used after the episode ends.

            REWARD RULES:
                1. To-do """
        ## APPLY REWARDS
        rewards = []

        # We apply the rewards in this loop. Ignore last position since we don't have the next value.
        for i in range(len(self.ep_memory)-1):
            # Reward
            price = self.ep_memory[i][1]
            next_price = self.ep_memory[i+1][1]
            change =  (next_price / price) - price # % profit/loss

            # Calculate basic reward
            action = self.ep_memory[i][2]
            reward = 0
            if action == [1,0]:
                reward = change * 100
            else:
                reward = change * -100

            # Apply punishment if a trade is made (to avoid over-trading)
            if action != self.ep_memory[i+1][2]:
                reward -= 0.25

            rewards.append(reward)

        ## DISCOUNT REWARDS
        # Get empty array with the same size as the rewards array
        discounted_rewards = np.zeros_like(rewards)
        # Variable that stores value of the discounted reward being calculated by the loop
        current_reward = 0.0
        ###########################################
        # MAY NEED TO REVERSE THE FOLLOWING RANGE #
        ###########################################
        # Loop that does the magic
        for i in reversed(range(len(rewards))):
            # Calculate the discounted reward
            current_reward = current_reward * gamma + rewards[i]
            # Store it in the array
            discounted_rewards[i] = current_reward

        ## NORMALIZE REWARDS
        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards)
        discounted_normalized_rewards = (discounted_rewards - mean) / (std)

        return np.array(discounted_normalized_rewards)


    def reset(self):
        """ Resets step-dependent variables """
        self.stack = deque(df.price.values[0:self.current_step], maxlen=stack_size)
        self.current_step = stack_size
        self.bought = 0
        self.ep_memory = []

        state = np.array([self.stack])

        # Add to step
        self.current_step += 1

        return state


env = Environment(df)

action_size = len(possible_actions)

## TRAINING Hyperparameters
max_episodes = 300
learning_rate = 0.001

""" Network """
tf.reset_default_graph()
with tf.name_scope("inputs"):
    states_ = tf.placeholder(tf.float32, [None, stack_size], name="states_")
    actions_ = tf.placeholder(tf.int32, [None, action_size], name="actions_")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards_")

    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(states_, 10, tf.nn.relu)

    with tf.name_scope("fc2"):
        fc2 = tf.layers.dense(fc1, action_size, tf.nn.relu)

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc2)

    with tf.name_scope("loss"):
        # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
        # If you have single-class labels, where an object can only belong to one class, you might now consider using
        # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc2, labels = actions_)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)

    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

""" Train """
allRewards = []
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]

# Visualization
losses = []
rewards = []

# Avoid crashes
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    for episode in range(max_episodes):
        episode_rewards_sum = 0

        # Launch the game
        state = env.reset()

        while True:
            # Choose action a. Get softmax probability for every possible action.
            action_probability_distribution = sess.run(action_distribution, feed_dict={states_: state})

            # Select action based on the the actions probability distribution . Ravel() flattens the array (2D -> 1D).
            action = np.random.choice(range(action_size), p=action_probability_distribution.ravel())
            action = possible_actions[action]

            # Perform action & get next data
            new_state, done = env.step(action)

            # Store s, a
            episode_states.append(state)
            episode_actions.append(action)

            # Once the whole episode is done we can train the network
            if done:
                # Calculate discounted rewards
                discounted_rewards = env.calculate_rewards()

                # Calculate sum of rewards
                episode_rewards_sum = discounted_rewards[-1]

                # Append the reward of the episode to allRewards so we can visualize the progress.
                allRewards.append(episode_rewards_sum)

                # Mean reward
                mean_reward = np.mean(allRewards)

                # Max reward
                maximumRewardRecorded = np.amax(allRewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Mean Reward", mean_reward)
                print("Max reward so far: ", maximumRewardRecorded)
                np.vstack(np.array(episode_actions)[:-1]).shape
                len(env.ep_memory)


                # Feedforward, gradient and backpropagation.
                # Loss: the softmax_cross_entropy between the results from the last dense layer vs the onehot-encoded actions
                loss_, _ = sess.run([loss, train_opt], feed_dict={states_: np.vstack(np.array(episode_states[:-1])),
                                                                  actions_: np.vstack(np.array(episode_actions[:-1])),
                                                                  discounted_episode_rewards_: discounted_rewards})

                print("Loss: ", loss_)
                losses.append(loss_)
                rewards.append(discounted_rewards)

                # Reset the transition stores
                episode_states, episode_actions, episode_rewards = [],[],[]

                break

            state = new_state

plt.plot(rewards[290])
plt.plot(losses)
