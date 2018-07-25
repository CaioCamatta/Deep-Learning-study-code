import tensorflow as tf
import numpy as np
import gym
import matplotlib.pylab as plt
from talib.abstract import *
import pandas as pd
from collections import deque
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Learning/Stock/data/data0.csv')
data = data[['open', 'high', 'low', 'close', 'volume']]

# WHAT MIGHT BE WRONG:
#   WHEN TRADING IT DOESNT KNOW IF IT HAS BOUGHT PREVIOUSLY, SO MAYBE IT WILL SIGNAL 1 BUT THERE'S NOTHING TO SELL
#   OVER TRADING
#   SELLING SHOULD REWARD IF PRICE DROPS
#   FINAL BALANCE SHOULD INFLUENCE OVERALL REWARD

# PARAMETERS
action_size = 2
interval_size = 20
state_size = (interval_size, 13) # interval_size x number of columns (OHLC, V, indicators...)
max_epochs = 1000
learning_rate = 0.0005
gamma = 0.999

class Environment:
    """ To run an entire epoch do:
            1. env.step() many times
            2. env.get_rewards()
            3. env.reset() """
    def __init__(self, dataframe):
        # Data that can be reset
        self.current_step = 0          # initial step
        self.balance = 100
        self.buy_price = None                     # Price in which the bitcoin was bought
        self.prev_action = None
        self.ep_trade_memory = []

        # Non-resetable
        self.data = dataframe             # holds the dataframe
        self.data_len = len(dataframe)    # length of the dataframe
        self.n_of_episodes = self.data_len // interval_size # Number of episodes in an epoch


    def step(self, action):
        """ Return next state and whether the dataframe is finished or not (done) """
        done = False

        # Done = true if we ran out of data.
        if self.current_step == self.n_of_episodes:
            done = True
            print('Done: True')

        # Define state
        index = self.current_step*interval_size
        state = self.data[index:index+interval_size, :]

        # Current price is the last close from the current interval
        current_price = self.data[index-1,3]

        # Buy current action is 1 (in) and prev_action is 0 (out)
        if action==1 and self.prev_action==0:
            self.buy(current_price)
            self.ep_trade_memory.append((self.current_step, self.balance, "BUY"))

        # Sell if current action is 0 (out) and prev_action is 1 (in) and if we had bought previously
        elif action==0 and self.prev_action==1 and self.buy_price!=None:
            self.sell(current_price)
            self.ep_trade_memory.append((self.current_step, self.balance, "SELL"))

        # Add to step
        self.current_step += 1
        self.prev_action = action

        return state, done

    def sell(self, current_price):
        """ Sells and update the balance """
        # Calculate new balance
        self.balance *= (self.buy_price / current_price)*0.997
        self.buy_price = None

    def buy(self, current_price):
        """ Buys """
        self.buy_price = current_price

    def reset(self):
        """ Resets Environment """
        self.current_step = 0
        self.balance = 100
        self.buy_price = None
        self.prev_action = None
        self.ep_trade_memory = []

        # Define state
        index = self.current_step*interval_size
        state = self.data[index:index+interval_size, :]

        # Add to step
        self.current_step += 1

        return state

    def get_rewards(self):
        """ Calculates rewards for the epoch.
                Reward for doing nothing is -0.05.
                Reward for each trade is profit or loss * 50 + 0.05 (to promote trading)
            OBS: the reward is the same from when it buys until it sells."""
        # Creates grid of reward
        clean_rewards = [0 for i in range(self.n_of_episodes)]

        # Substitute values in the grid to trade rewards.
        # [2*i for i in range(len(self.ep_trade_memory)//2)]
        for i in range(len(self.ep_trade_memory)-1):
            reward = ((self.ep_trade_memory[i+1][1] / self.ep_trade_memory[i][1])-1)*env.balance*0.997
            clean_rewards[self.ep_trade_memory[i][0]:self.ep_trade_memory[i+1][0]] = [reward for i in range(self.ep_trade_memory[i+1][0]-self.ep_trade_memory[i][0])]

        return clean_rewards

class Helper:
    def __init__(self):
        self.a = 1

    def apply_indicators(self, input_data):
        """ Applies indicator data to every entry on dataframe. Removes the initial
        rows than have nan values due to the indicators."""
        inputs = {
            'open': input_data[:,0],
            'high': input_data[:,1],
            'low': input_data[:,2],
            'close': input_data[:,3],
            'volume': input_data[:,4],
       }

       # Calculate indicators
        ema = EMA(inputs)
        _, _, macd = MACD(inputs)
        bbands = BBANDS(inputs)
        bbands = bbands[0]-bbands[1]
        rsi = RSI(inputs)/100
        mom = MOM(inputs)
        atr = ATR(inputs)
        adx = ADX(inputs)/100
        ultosc = ULTOSC(inputs)/100

        # Put everthing together and delete NaN values
        final_column = np.column_stack([input_data, ema, macd, bbands, rsi, mom, atr, adx, ultosc])
        final_column = final_column[~np.isnan(final_column).any(axis=1)]

        # Scale indicators (except the ones that are already good)
        sc = MinMaxScaler(feature_range=(1,2))
        final_column[:,6] = sc.fit_transform(final_column[:,6].reshape(-1,1)).reshape(final_column.shape[0])
        sc = MinMaxScaler(feature_range=(1,2))
        final_column[:,7] = sc.fit_transform(final_column[:,7].reshape(-1,1)).reshape(final_column.shape[0])
        sc = MinMaxScaler(feature_range=(1,2))
        final_column[:,8] = sc.fit_transform(final_column[:,8].reshape(-1,1)).reshape(final_column.shape[0])
        sc = MinMaxScaler(feature_range=(1,2))
        final_column[:,9] = sc.fit_transform(final_column[:,9].reshape(-1,1)).reshape(final_column.shape[0])
        sc = MinMaxScaler(feature_range=(1,2))
        final_column[:,10] = sc.fit_transform(final_column[:,10].reshape(-1,1)).reshape(final_column.shape[0])
        sc = MinMaxScaler(feature_range=(1,2))
        final_column[:,11] = sc.fit_transform(final_column[:,11].reshape(-1,1)).reshape(final_column.shape[0])
        sc = MinMaxScaler(feature_range=(1,2))
        final_column[:,12] = sc.fit_transform(final_column[:,12].reshape(-1,1)).reshape(final_column.shape[0])

        return final_column

    def scale_data(self, input_data):
        """ Scales data to range 1-2 """
        # Normalize currency values ('open', 'high', 'low', 'close')
        sc_value = MinMaxScaler(feature_range=(1,2))
        sc_value.fit(np.array(input_data.iloc[:, :4]).reshape(input_data.shape[0]*4).reshape(-1,1))
        input_data[['open', 'high', 'low', 'close']] =sc_value.transform(input_data.iloc[:, :4])

        # Normalize volume
        sc_volume = MinMaxScaler(feature_range=(1,2))
        input_data['volume'] = sc_volume.fit_transform(np.array(input_data.iloc[:, :4]))

        return input_data

    def discount_and_normalize_rewards(self, epoch_rewards):
        """ Returns list of discounted rewards. Rewards closer at the beginning are more
            important so they are very high. The last reward is equal to 1 (before normalizing)
            so the first reward has a huge value (before normalizing). Try printing it to see."""
        # Get empty array with the same size as the rewards array
        discounted_epoch_rewards = np.zeros_like(epoch_rewards)

        # Variable that stores value of the discounted reward being calculated by the loop
        current_reward = 0.0
        # Loop that does the magic
        for i in reversed(range(len(epoch_rewards))):
            # Calculate the discounted reward
            current_reward = current_reward * gamma + epoch_rewards[i]
            # Store it in the array
            discounted_epoch_rewards[i] = current_reward

        # Normalize.
        mean = np.mean(discounted_epoch_rewards)
        std = np.std(discounted_epoch_rewards)
        discounted_epoch_rewards = (discounted_epoch_rewards - mean) / (std)

        return discounted_epoch_rewards


helper = Helper()
data = helper.scale_data(data)
data = helper.apply_indicators(np.array(data))

env = Environment(data)

""" Network """
tf.reset_default_graph()
with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, *state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_epoch_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_epoch_rewards")

    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(inputs = input_ , units = 256, activation=tf.nn.relu)

    with tf.name_scope("fc2"):
        fc2 = tf.layers.dense(inputs = fc1, units = 128, activation=tf.nn.relu)

    with tf.name_scope("fc3"):
        fc3 = tf.layers.dense(inputs = fc2, units = action_size, activation= None)

    with tf.name_scope("softmax"):
        action_distribution = tf.reduce_mean(tf.nn.softmax(fc3), 1)

    with tf.name_scope("loss"):
        # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
        # If you have single-class labels, where an object can only belong to one class, you might now consider using
        # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
        logits = tf.reduce_mean(fc3, 1)
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_epoch_rewards_)

    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

""" Tensorboard """
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("E:/Code/Anaconda/Learning/RL/Stock/tensorboard/pg-2")
## Losses
tf.summary.scalar("Loss", loss)
## Reward mean
tf.summary.scalar("Reward_mean", mean_reward_)
# Operation
write_op = tf.summary.merge_all()

""" Train """
avg_rewards = []
maximumRewardRecorded = 0
epoch = 0
epoch_states, epoch_actions = [],[]
balance_history = []
losses = []
correct_actions = []
repeated_action_one = []                  # How many times did we sell per epoch ?

env = Environment(data)

# Avoid crashes
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(max_epochs):
        epoch_rewards_sum = 0

        # Launch the game
        state = env.reset()

        while True:
            # Choose action a. Get softmax probability for every possible action.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape(1, *state_size)})

            # Select action based on the the actions probability distribution . Ravel() flattens the array (2D -> 1D).
            action = np.random.choice(range(action_size), p=action_probability_distribution.ravel())

            # Perform action & get next data
            new_state, done = env.step(action)

            # Store s, a, r
            epoch_states.append(list(state))

            # In order to pass the action_ mask placeholder we must first one hot enconde the action,
            #   since the 'action' is not onehot encoded yet. PS: [1,0]=left and [0,1]=right
            action_ = np.zeros(action_size)
            action_[action] = 1
            epoch_actions.append(list(action_))

            # Once the whole epoch is done we can train the network
            if done:
                # Get rewards
                epoch_rewards = env.get_rewards()

                # Calculate sum of rewards
                epoch_rewards_sum = np.sum(epoch_rewards)

                # Append the sum of reward of the epoch to avg_rewards so we can visualize the progress.
                avg_rewards.append(epoch_rewards_sum)

                # Mean reward
                mean_reward = np.mean(avg_rewards)

                # Max reward
                maximumRewardRecorded = np.amax(avg_rewards)

                # Number of correct predictions
                correct_actions_epoch = (np.array(epoch_rewards)>0).sum()
                correct_actions.append(correct_actions_epoch)

                # Get balance
                balance = env.balance
                balance_history.append(balance)

                if epoch % 5 == 0:
                    print("==========================================")
                    print("Epoch: ", epoch)
                    print(f"Correct Predictions: {correct_actions_epoch} / {env.n_of_episodes}")
                    print("Mean Reward", mean_reward)
                    print("Max reward so far: ", maximumRewardRecorded)
                    print("Balance: ", balance)

                # Calculate discounted reward
                discounted_epoch_rewards = helper.discount_and_normalize_rewards(epoch_rewards)

                # Feedforward, gradient and backpropagation.
                # Loss: the softmax_cross_entropy between the results from the last dense layer vs the onehot-encoded actions
                loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.array(epoch_states),
                                                                  actions: np.array(epoch_actions),
                                                                  discounted_epoch_rewards_: discounted_epoch_rewards})

                losses.append(loss_)

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={input_: np.array(epoch_states),
                                                        actions: np.array(epoch_actions),
                                                        discounted_epoch_rewards_: discounted_epoch_rewards,
                                                        mean_reward_: mean_reward})

                writer.add_summary(summary, epoch)

                # Save the number of action 1. For debugging
                repeated_action_one.append(len([i for i in epoch_actions if i==[0,1]]))

                # Reset the transition stores
                if epoch != (max_epochs-1):
                    epoch_states, epoch_actions = [],[]

                break

            state = new_state

# PLOT BUY / SELL
plt.plot(data[:, 3][2500:2600], '-gD', markevery=[i for i, n in enumerate(epoch_actions[2500:2600]) if n[1]==1])
plt.plot(data[:, 3][2500:2600], '-rD', markevery=[i for i, n in enumerate(epoch_actions[2500:2600]) if n[1]==0])

# PLOT PROGRESSION
plt.plot(losses)
plt.plot([i/1000 for i in balance_history])
plt.plot([i/10000 for i in correct_actions])
