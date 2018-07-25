import numpy as np
from matplotlib import animation
from IPython.display import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import gym
env = gym.make("CartPole-v0")

# Discounted rewards
def discount_rewards(r, gamma=0.99):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
    """
    return np.array([val * (gamma ** i) for i, val in enumerate(r)])

# Constants defining our neural network
hidden_layer_neurons = 10
batch_size = 25
learning_rate = 1e-2
gamma = .99
state_space = 4

tf.reset_default_graph()

# Define input placeholder
state_ph = tf.placeholder(tf.float32, [None, state_space], name="input_x")

# First layer of weights (10 neurons)
W1 = tf.get_variable("W1", shape=[state_space, hidden_layer_neurons],
                    initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(state_ph,W1))

# Second layer of weights (1 output neuron)
W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, 1],
                    initializer=tf.contrib.layers.xavier_initializer())
output = tf.nn.sigmoid(tf.matmul(layer1,W2)) # shape [50, 1], contains probabilities

# We need to define the parts of the network needed for learning a policy
trainable_vars = [W1, W2]
input_y = tf.placeholder(tf.float32, [None,1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal") # discounted rewards

# Loss function
#   loglik = tf.log(input_y*(input_y — probability) + (1 — input_y)*(input_y + probability))
# If input_y is 0, then the first term is eliminated, and it becomes tf.log(probability) .
# If input_y is 1 instead then it becomes tf.log(1-probability) and we ensure that the term inside the log is never negative, and that we can maximally utilize probabilities related to both of the actions.
# Below is equivalent to: 0 if input_y (probability) == output_p (probability) else 1
log_lik = tf.log(input_y * (input_y - output) + (1 - input_y) * (input_y + output))
loss = -tf.reduce_mean(log_lik * advantages)

# Gradients
#   The gradBuffer is actually being used to collect the gradients together before applying them to the policy. This is done in order to reduce the variance of the gradients before they are applied. If we used a high-variance set of gradients the policy might become destabilized
# Use one gradient for each possible action
new_grads = tf.gradients(loss, trainable_vars)
W1_grad = tf.placeholder(tf.float32, name="batch_grad1")
W2_grad = tf.placeholder(tf.float32, name="batch_grad2")

# Learning
batch_grad = [W1_grad, W2_grad]
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_grads = adam.apply_gradients(zip(batch_grad, [W1, W2]))


####### Trainin #######
reward_sum = 0
init = tf.global_variables_initializer()

# Placeholders for our observations, outputs and rewards
xs = np.empty(0).reshape(0,state_space)
ys = np.empty(0).reshape(0,1)
rewards = np.empty(0).reshape(0,1)

# Setting up our environment
sess = tf.Session()
rendering = False
sess.run(init)
observation = env.reset()

# Placeholder for out gradients
gradients = np.array([np.zeros(var.get_shape()) for var in trainable_vars])

num_episodes = 5000
num_episode = 0

while num_episode < num_episodes:
    # Append the observations to our batch
    x = np.reshape(observation, [1, state_space])

    # Run the neural net to determine output
    tf_prob = sess.run(output, feed_dict={state_ph: x})

    # Determine the output based on our net, allowing for some randomness
    y = 0 if tf_prob > np.random.uniform() else 1

    # Append the observations and outputs for learning
    xs = np.vstack([xs, x])
    ys = np.vstack([ys, y])

    # Determine the oucome of our action
    observation, reward, done, _ = env.step(y)
    reward_sum += reward
    rewards = np.vstack([rewards, reward])

    if done:
        # Determine standardized rewards
        discounted_rewards = discount_rewards(rewards, gamma)
        discounted_rewards -= discounted_rewards.mean()
        discounted_rewards /= discounted_rewards.std()

        # Append gradients for case to running gradients
        gradients += np.array(sess.run(new_grads, feed_dict={state_ph: xs,
                                               input_y: ys,
                                               advantages: discounted_rewards}))

        # Clear out game variables
        xs = np.empty(0).reshape(0,state_space)
        ys = np.empty(0).reshape(0,1)
        rewards = np.empty(0).reshape(0,1)

        # Once batch full
        if num_episode % batch_size == 0:
            # Updated gradients
            sess.run(update_grads, feed_dict={W1_grad: gradients[0],
                                             W2_grad: gradients[1]})
            # Clear out gradients
            gradients *= 0

            # Print status
            print("Average reward for episode {}: {}".format(num_episode, reward_sum/batch_size))

            if reward_sum / batch_size > 200:
                print("Solved in {} episodes!".format(num_episode))
                break
            reward_sum = 0
        num_episode += 1
        observation = env.reset()



####### See our trained bot in action #######
observation = env.reset()
observation
reward_sum = 0

while True:
    env.render()

    x = np.reshape(observation, [1, state_space])
    y = sess.run(output, feed_dict={state_ph: x})
    y = 0 if y > 0.5 else 1
    observation, reward, done, _ = env.step(y)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        env.close()
        break
