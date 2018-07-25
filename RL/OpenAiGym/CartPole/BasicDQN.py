import gym
import random
import numpy as np
import tensorflow as tf
from statistics import mean, median
from collections import Counter
import uuid

env = gym.make('CartPole-v0')
env.reset()
possible_actions = [[1,0], [0,1]]

# PARAMETERS
goal_steps = 500
score_requirement = 80 # only learn from games that achieve at least 50 score/steps.
initial_games = 10000
possible_actions = [[1,0], [0,1]]
n_of_trains = 1000
batch_size = 32
gamma = 0.99
learning_rate = 0.0002
rate_of_update = 0.001

# Random games just for visualization
def some_random_games_first():
    """ Run some random games just to understand """
    for episode in range(5):
        env.reset()

        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample() # Generate random action
            observation, reward, done, info = env.step(action)

            if done:
                break

    env.close()

# Grab training data
def initial_population():
    """ This function will run 'initial_games' games with random actions, and if
        the score in these games is greater than 'score_requirement', it will save
        the memory of the game ([observation[t-1] ,action[t], ...]) so we can later
        use it teach a Neural Network."""
    training_data = [] # observation and moves (random) made
    scores = []
    accepted_scores = [] # scores above 50

    # For every game
    for _ in range(initial_games):
        score = 0
        game_memory = [] # Store the memory for the game to use it in case we get an acceptable score

        prev_observation = [] # temporarely stores observation

        # For every step
        for _ in range(goal_steps):
            action = random.randrange(0,2) # generate random action
            observation, reward, done, info = env.step(action) # register environment info

            # if we are not in the first step (i.e. we've already made an observation):
            # Remember: an empty list returns False
            if len(prev_observation) > 0:
                # observation occurs after the action, so save previous_observation and current action.
                game_memory.append([prev_observation, possible_actions[action], reward, observation, done])

            prev_observation = observation
            score += reward

            if done:
                break

        # Save game data to training data if the score is acceptable
        if score >= score_requirement:
            accepted_scores.append(score)

            for data in game_memory:
                training_data.append(data)

        env.reset()
        scores.append(score)

    # Save data
    #np.save('saved.npy', np.array(training_data))

    # Print results
    print('Average accepted scores: ', mean(accepted_scores))
    print('Median accepted scores: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

class Neural_network_model:
    def __init__(self):
        # Placeholders
        self.inputs_ = tf.placeholder(tf.float32, [None, 4], name="inputs")
        self.actions_ = tf.placeholder(tf.float32, [None, 2], name="actions_")

        # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
        self.target_Q = tf.placeholder(tf.float32, [None], name="target")

        self.fc = tf.layers.dense(inputs = self.inputs_,
                                  units = 10,
                                  activation = tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.fc2 = tf.layers.dense(inputs = self.fc,
                                  units = 2,
                                  activation = tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output = tf.layers.dense(inputs = self.fc2,
                                      units = 2,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      activation=None)


        # Q is our predicted Q value for the specific chosen action (that's why we multiply)
        # Reduce sum: [0.642, 0] > 0.642
        # A much better strategy is to have the network multiply its outputs by a “mask” corresponding to the one-hot encoded action, which will set all its outputs to 0 except the one for the action we actually saw. We can then pass 0 as the target to for all unknown actions and our neural network should thus perform fine. When we want to predict for all actions, we can then simply pass a mask of all 1s.
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

        # The loss is the difference between our predicted Q_values and the Q_target
        # Sum(Qtarget - Q)^2
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

def updateTargetNetworkOp():
    """ This function creates the operation that copies the variables from the model to the target network. """
    tfVars = tf.trainable_variables()
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*rate_of_update) + ((1-rate_of_update)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(sess, op_holder):
    """ Runs the operations that copy variables from secondary to primary network """
    for op in op_holder:
        sess.run(op)

# Basics
tf.reset_default_graph()

training_data = initial_population()

model = Neural_network_model()
target_model = Neural_network_model()

op_holder = updateTargetNetworkOp()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75) # avoid crashing
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    for train in range(n_of_trains):
        # Get random batch
        batch = [training_data[i] for i in np.random.choice(np.arange(len(training_data)),size = batch_size,replace = False)]

        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])

        Qs = sess.run(model.Q, feed_dict = {model.inputs_: states,
                                            model.actions_: actions})
        next_Qs = sess.run(target_model.Q, feed_dict = {target_model.inputs_: next_states,
                                                        target_model.actions_: actions})

        target_Qs = []
        for i in range(batch_size):
            if dones[i] == True:
                target_Qs.append(rewards[i])
            else:
                target_Qs.append(rewards[i] + gamma * np.amax(next_Qs[i]))

        # Remember : will run Qs again
        loss, _ = sess.run([model.loss, model.optimizer], feed_dict={model.inputs_: states,
                                                                     model.actions_: actions,
                                                                     model.target_Q: target_Qs})

        updateTarget(sess, op_holder)

        if train % 30 == 0:
            print(f'Train number {train}, Loss: {loss}')

    for each_game in range(30):
        scores = []
        score = 0
        game_memory = []
        prev_observation = []

        env.reset()

        for _ in range(300):
            env.render()

            if len(prev_observation) == 0: # if there's nothing
                action = env.action_space.sample()
            else:
                output = sess.run(model.output, feed_dict = {model.inputs_: prev_observation.reshape(1, 4)})
                action = np.argmax(output)

            new_observation, reward, done, info = env.step(action)
            prev_observation = new_observation

            score += reward
            if done:
                break

    env.close()
