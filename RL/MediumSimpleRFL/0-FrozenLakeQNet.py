# Tutorial available at: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

#   Networks are a necessity since tables don't scale. We need to describe our
# state and and produce Q values for actions without a table. With Networks we
# take multiple possible states (represented by a vector) and learn to map them
# to Q values.

#   In this example, we will be using a one-layer network which takes the state
# encoded in a one-hot vector (1x16), and produces a vector of 4 Q-values (1/action)

#    The loss function will be sum-of-squares, where the difference between the
# current predicted Q-values, and the “target” value is computed and the
# gradients passed through the network.
#   The Q in "Loss = ∑(Q-target - Q)²" is equal to "Q(s,a) = r + γ(max(Q(s’,a’))"

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

##### NETWORK #####
tf.reset_default_graph()

# These lines establish the feed-forward part of the network used to choose actions
# OBS: [1,16] * [16,4] = [1,4]
state_ph = tf.placeholder(shape=[1,16],dtype=tf.float32) # Current state 16 possibilities
W = tf.Variable(tf.random_uniform([16,4],0,0.01)) # 4 output weights
Qout = tf.matmul(state_ph,W) # Multiply current state by weights
predict = tf.argmax(Qout,1) # Predict action (only thing that's not one hot encoded)

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32) # Target actions (also one hot encoded)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

##### TRAINING #####
init = tf.global_variables_initializer()

# Set learning parameters
y = .99
e = 0.1 # chance of random action
num_episodes = 10

#create lists to contain total rewards and steps per episode
jList = []
rList = []

with tf.Session() as sess:
    sess.run(init)

    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0 # total reward achieve by the episode
        done = False # done
        j = 0 # step

        #The Q-Network
        while j < 99:
            j+=1

            # Choose an action by greedily (with e chance of random action) from the Q-network
            # PS: 'np.identity(16)[s:s+1]'' is a onehot-encoded version of the position in the game matrix
            a, Q = sess.run([predict, Qout],feed_dict={state_ph:np.identity(16)[s:s+1]})

            # 10% Chance of running a random action
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            #Get new state and reward from environment
            s_, r, done, _ = env.step(a[0])

            # Obtain the Q_ values by feeding the new state through our network
            Q_ = sess.run(Qout,feed_dict={state_ph:np.identity(16)[s_:s_+1]})

            # Obtain maxQ' and set our target value for chosen action.
            maxQ_ = np.max(Q_)
            targetQ = Q
            targetQ[0,a[0]] = r + y*maxQ_

            #Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel,W],feed_dict={state_ph:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s_

            if done == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break

        jList.append(j)
        rList.append(rAll)

print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

plt.plot(rList)
plt.plot(jList)
