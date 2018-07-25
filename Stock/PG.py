import tensorflow as tf
import numpy as np
import gym
import matplotlib.pylab as plt
import pandas as pd
from collections import deque

action_size = 2
stack_size = 20
state_size = stack_size

## MAKE DATA
data_points = 401
x = np.linspace(0, 10*np.pi, data_points)
df = pd.DataFrame({'price':np.power(np.sin(x)+2,1/1.6)})

## TRAINING Hyperparameters
max_episodes = 1500
learning_rate = 0.001
gamma = 0.995

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


    def step(self, action):
        """ Return next state and whether the dataframe is finished or not (done) """
        done = False

        # Done = true if we ran out of data.
        if self.current_step == self.df_len-2:
            done = True

        # Append current price to stack
        current_price = self.df.price.values[self.current_step]
        self.stack.append(current_price)

        # Define state
        state = np.array([self.stack])

        # Reward
        next_price = self.df.price.values[self.current_step+1]
        change =  (next_price / current_price) - 1 # % profit/loss

        # Calculate basic reward
        if action == 1:
            reward = change * 100
        else:
            reward = change * -100

        # Apply punishment if a trade is made (to avoid over-trading)

        # Add to step
        self.current_step += 1

        return state, reward, done

    def reset(self):
        """ Resets step-dependent variables """
        self.stack = deque(df.price.values[0:self.current_step], maxlen=stack_size)
        self.current_step = stack_size
        self.bought = 0

        state = np.array([self.stack])

        # Add to step
        self.current_step += 1

        return state

def discount_and_normalize_rewards(episode_rewards):
    """ Returns list of discounted rewards. Rewards closer at the beginning are more
        important so they are very high. The last reward is equal to 1 (before normalizing)
        so the first reward has a huge value (before normalizing). Try printing it to see."""
    # Get empty array with the same size as the rewards array
    discounted_episode_rewards = np.zeros_like(episode_rewards)

    # Variable that stores value of the discounted reward being calculated by the loop
    current_reward = 0.0
    # Loop that does the magic
    for i in reversed(range(len(episode_rewards))):
        # Calculate the discounted reward
        current_reward = current_reward * gamma + episode_rewards[i]
        # Store it in the array
        discounted_episode_rewards[i] = current_reward

    # Normalize.
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards

""" Network """
tf.reset_default_graph()
with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")

    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(inputs = input_,units = 64, activation=tf.nn.relu)

    with tf.name_scope("fc2"):
        fc2 = tf.layers.dense(inputs = fc1, units = 96, activation=tf.nn.relu)

    with tf.name_scope("fc3"):
        fc3 = tf.layers.dense(inputs = fc2, units = 128, activation=tf.nn.relu)

    with tf.name_scope("fc4"):
        fc4 = tf.layers.dense(inputs = fc3, units = action_size, activation= None)

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc4)

    with tf.name_scope("loss"):
        # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
        # If you have single-class labels, where an object can only belong to one class, you might now consider using
        # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc4, labels = actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)

    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

""" Tensorboard """
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("E:/Code/Anaconda/Learning/RL/Stock/tensorboard/pg")
## Losses
tf.summary.scalar("Loss", loss)
## Reward mean
tf.summary.scalar("Reward_mean", mean_reward_)
# Operation
write_op = tf.summary.merge_all()

""" Train """
allRewards = []
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]
losses = []
correct_actions = []

env = Environment(df)

# Avoid crashes
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    for episode in range(max_episodes):
        episode_rewards_sum = 0

        # Launch the game
        state = env.reset()

        while True:
            # Choose action a. Get softmax probability for every possible action.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state})

            # Select action based on the the actions probability distribution . Ravel() flattens the array (2D -> 1D).
            action = np.random.choice(range(action_size), p=action_probability_distribution.ravel())

            # Perform action & get next data
            new_state, reward, done = env.step(action)

            # Store s, a, r
            episode_states.append(list(state))
            episode_rewards.append(reward)

            # In order to pass the action_ mask placeholder we must first one hot enconde the action,
            #   since the 'action' is not onehot encoded yet. PS: [1,0]=left and [0,1]=right
            action_ = np.zeros(action_size)
            action_[action] = 1
            episode_actions.append(list(action_))

            # Once the whole episode is done we can train the network
            if done:
                # Calculate sum of rewards
                episode_rewards_sum = np.sum(episode_rewards)

                # Append the reward of the episode to allRewards so we can visualize the progress.
                allRewards.append(episode_rewards_sum)

                # Mean reward
                mean_reward = np.mean(allRewards)

                # Max reward
                maximumRewardRecorded = np.amax(allRewards)

                # Number of correct predictions
                correct_actions_episode = (np.array(episode_rewards)>0).sum()
                correct_actions.append(correct_actions_episode)


                print("==========================================")
                print("Episode: ", episode)
                print(f"Correct Predictions: {correct_actions_episode} / 379")
                print("Mean Reward", mean_reward)
                print("Max reward so far: ", maximumRewardRecorded)

                # Calculate discounted reward
                discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)

                # Feedforward, gradient and backpropagation.
                # Loss: the softmax_cross_entropy between the results from the last dense layer vs the onehot-encoded actions
                loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                  actions: np.vstack(np.array(episode_actions)),
                                                                  discounted_episode_rewards_: discounted_episode_rewards})

                losses.append(loss_)

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={input_: np.vstack(np.array(episode_states)),
                                                        actions: np.vstack(np.array(episode_actions)),
                                                        discounted_episode_rewards_: discounted_episode_rewards,
                                                        mean_reward_: mean_reward})

                writer.add_summary(summary, episode)

                # Reset the transition stores
                if episode != 1499:
                    episode_states, episode_actions, episode_rewards = [],[],[]

                break

            state = new_state

plt.plot(df, '-gD', markevery=[i for i, n in enumerate(episode_actions) if n[1]==0])
plt.plot(losses)
plt.plot(correct_actions)
