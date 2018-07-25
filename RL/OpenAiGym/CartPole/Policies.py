import random
import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v1')

###### Create function to run a game ######
def run_episode(env, parameters, render=False):
    observation = env.reset()
    totalreward = 0

    for _ in range(1000):
        if render:
            env.render()

        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward

        if done:
            env.close()
            break

    return totalreward


###### Random search and return best paramaters ######
# keep trying random weights, and pick the one that performs the best.
def randomSearch():
    bestparams = None
    bestreward = 0

    for index in range(10000):
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env,parameters)

        if reward > bestreward:
            print('Better at: ', index)
            bestreward = reward
            bestparams = parameters

        # considered solved if the agent lasts 500 timesteps
        if reward >= 500:
            print('Reached')
            break

    return bestparams

best_parameters = randomSearch()
print(best_parameters)
run_episode(env, best_parameters, render=True)


###### Hill Climbing ######
# We start with some randomly chosen initial weights. Every episode, add some
# noise to the weights, and keep the new weights if the agent improves.
def hillClimbing():
    bestparams = None
    bestreward = 0
    noise_scaling = 0.1 # Noise added
    parameters = np.random.rand(4) * 2 - 1

    for index in range(10000):
        updated_params = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling
        reward = run_episode(env,parameters)

        if reward > bestreward:
            print('Better at: ', index)
            bestreward = reward
            bestparams = parameters

        # considered solved if the agent lasts 500 timesteps
        if reward >= 500:
            print('Reached')
            break

    return bestparams

best_parameters = randomSearch()
print(best_parameters)
run_episode(env, best_parameters, render=True)

###### Policy Gradient ######
def policy_gradient():
    """ Updates parameters in order to make it more likely to choose a prefered
        (correct) action for each state """
    # Placeholders for state and prefered action
    state = tf.placeholder("state",[None,4])
    actions = tf.placeholder("float",[None,2])
    # Variable for parameters 4 inputs, 2 output(one-hot)
    params = tf.get_variable("policy_parameters",[4,2])
    # Layer
    y = tf.matmul(state,params)
    # Output
    probabilities = tf.nn.softmax(y)
    # Multiply probabilitie for each action by the prefered action.
    good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions), axis=[1])
    # maximize the log probability
    log_probabilities = tf.log(good_probabilities)
    loss = -tf.reduce_sum(log_probabilities)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

def value_gradient():
    """ We a define some value for each state, that contains the average return
        starting from that state"""
    # sess.run(calculated) to calculate value of state
    state = tf.placeholder("float",[None,4])
    w1 = tf.get_variable("w1",[4,10])
    b1 = tf.get_variable("b1",[10])
    h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
    w2 = tf.get_variable("w2",[10,1])
    b2 = tf.get_variable("b2",[1])
    calculated = tf.matmul(h1,w2) + b2

    # sess.run(optimizer) to update the value of a state
    newvals = tf.placeholder("float",[None,1])
    diffs = calculated - newvals
    loss = tf.nn.l2_loss(diffs)
    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

# tensorflow operations to compute probabilties for each action, given a state
pl_probabilities, pl_state = policy_gradient()
observation = env.reset()
actions = []
transitions = []
for _ in xrange(200):
    # calculate policy
    obs_vector = np.expand_dims(observation, axis=0) # [x] -> [[x]]
    probs = sess.run(pl_probabilities,feed_dict={pl_state: obs_vector})
    action = 0 if random.uniform(0,1) < probs[0][0] else 1

    # record the transition
    states.append(observation)
    actionblank = np.zeros(2)
    actionblank[action] = 1
    actions.append(actionblank)

    # take the action in the environment
    old_observation = observation
    observation, reward, done, info = env.step(action)
    transitions.append((old_observation, action, reward))
    totalreward += reward

    if done:
        break
