import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import gym                   # OpenAI Gym
import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames
from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs
from scipy import misc
import itertools as it

# LOSS EXPLANATION : one network calculates Q(s,a), and subsequently (using the same approximator, like the same NN), the value of Q(s',a'), in which (s',a') are one time-step later than (s,a). Using a perfect Q-function, the difference between Q(s,a) and Q(s',a') should be exactly the direct reward received in going from s to s' by action a: Q(s,a)=Q(s′,a′)+r. Therefore, δ is computed as δ=Q(s′,a′)+r−Q(s,a), neglecting discount rate γ.
env = gym.make('CartPole-v1')
possible_actions = [[1,0], [0,1]]

# HYPERPARAMETERS
state_size = env.observation_space.shape[0]      # Our input is a stack of 4 frames hence 96x96x4 (Width, height, channels)
action_size = env.action_space.n        # Actions space = list of possible actions
learning_rate =  0.0001                 # Alpha / learning rate
total_episodes = 300                    # Total episodes for training
max_steps = 10000                       # Max possible steps in an episode
batch_size = 2                         # Amount of arrays that will be fed into the network at once when training
rate_of_update = 0.001                  # Rate in which the target network will copy the primary network's variables
epsilon_start = 0.9                     # Exploration probability at start (epsilon greedy)
epsilon_stop = 0.1                    # Minimum exploration probability (epsilon greedy)
decay_rate = 0.003                      # Exponential decay rate for exploration prob (epsilon greedy)
gamma = 0.99                            # Q learning gamma (reward discount)
pretrain_epochs = 100                     # Initial train (to fill the memory)
memory_size = 20000                     # Max number of (s, a, r, s_, d) arrays stored in the memory
# Set training to false if you just want to see the model in action
training = True


""" Convenient functions are placed inside the helper class for better organization """
class Helper:
    def __init__(self):

        # Get list of trainable variables in both networks
        self.trainables = tf.trainable_variables()

        # Define operation that updates target network
        self.op_holder = self.updateTargetNetworkOp()

        # Start epsilon decay at 0 and start episilon
        self.decay_step = 0
        self.epsilon_probability = epsilon_start

    def preprocess_frame(self, frame):
        """ Preprocessing
            Take a frame.
            Resize it.
                __________________
                |                 |
                |                 |
                |                 |
                |                 |
                |_________________|

                to
                _____________
                |            |
                |            |
                |            |
                |____________|
            Normalize it.

            return preprocessed_frame """
        # Greyscale frame
        img = np.mean(frame,-1)

        # Remove black bar at the bottom
        img = img[:-12, :]

        # Resize
        img = misc.imresize(img, (resized_image_res))

        # Crop the screen (remove the roof because it contains no information) (not necessary here)
        cropped_frame = img

        # Normalize Pixel Values
        normalized_frame = cropped_frame/255.0

        return normalized_frame

    def stack_frames(self, state):
        """ Stacks frames so the AI can have a notion of movement """
        # Preprocess frame
        frame = self.preprocess_frame(state)

        # Append frame to deque, automatically removes the oldest frame
        self.stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(self.stacked_frames, axis=2)

        return stacked_state

    def updateTargetNetworkOp(self):
        """ This function creates the operation that copies the variables from the model to the target network. """
        tfVars = self.trainables
        total_vars = len(tfVars)
        op_holder = []
        for idx,var in enumerate(tfVars[0:total_vars//2]):
            op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*rate_of_update) + ((1-rate_of_update)*tfVars[idx+total_vars//2].value())))
        return op_holder

    def updateTarget(self, sess):
        """ Runs the operations that copy variables from secondary to primary network """
        for op in self.op_holder:
            sess.run(op)

    def epsilon_greedy(self):
        """ Epsilon Greedy. A True return should mean that the AI has to act randomly. """
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        random_number = np.random.rand()

        return self.epsilon_probability > random_number

    def reduceEpsilon(self):
        # Increase decay
        self.decay_step +=1

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        self.epsilon_probability = epsilon_stop + (epsilon_start - epsilon_stop) * np.exp(-decay_rate * self.decay_step)

    def getTargetQs(self, states, actions, rewards, next_states, dones, target_model, sess):
        # This list will hold the target Qs for the passes batch
        target_Qs_batch = []

        # Get Q values for next_state. Expected reward for each action
        target_Qs = sess.run(target_model.output, feed_dict = {target_model.inputs_: next_states})

        # Set Qhat = r if the episode ends at +1, otherwise set Qhat = r + gamma*maxQ(s', a')
        for i in range(batch_size):
            terminal = dones[i]

            # If we are in a terminal state, only equals reward
            if terminal:
                target_Qs_batch.append(rewards[i])
            else:
                target = rewards[i] + gamma * np.max(target_Qs[i])
                target_Qs_batch.append(target)

        # Put all targets inside of a nice array
        targets = np.array([each for each in target_Qs_batch])

        return targets


""" Experience Replay """
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        self.fill_randomly(pretrain_epochs)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False) # False so we get unique instances

        batch = [self.buffer[i] for i in index]

        # Separate batch
        states = np.array([each[0] for each in batch], ndmin=2)
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch], ndmin=2)
        dones = np.array([each[4] for each in batch])

        return states, actions, rewards, next_states, dones

    def fill_randomly(self, size):
        for i in range(size):# First we need a state
            state = env.reset()
            done = False

            while not done:
                # Random action
                action = env.action_space.sample()

                # Get the rewards
                observation, reward, done, _ = env.step(action)
                done = done

                #misc.imsave(f'E:/Code/Anaconda/Learning/RL/DQN/CarRacing-v0/imgs/img{i}.jpg', helper.preprocess_frame(observation))

                if done:
                    # We finished the episode
                    print("Done initial random fill")
                    next_state = np.zeros(state.shape)

                    # Add experience to memory
                    self.add((state, possible_actions[action], reward, next_state, done))

                    break

                else:
                    # Get the next state
                    next_state = observation

                    # Add experience to memory
                    self.add((state, possible_actions[action], reward, next_state, done))

                    # Our state is now the next_state
                    state = next_state


""" Define DQNetwork
        We take a stack of 4 frames as input
        It passes through 3 convnets
        Then it is flatened
        Finally it passes through 2 FC layers
        It outputs a Q value for each actions """
class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # Placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, (96-12)/2, 96/2, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.fc = tf.layers.dense(inputs = self.inputs_,
                                      units = 32,
                                      activation = tf.nn.relu,
                                      name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc,
                                      units = 64,
                                      activation = tf.nn.relu,
                                      name="fc2")

            self.output = tf.layers.dense(inputs = self.fc2,
                                          units = self.action_size,
                                          activation=None)

            # Q is our predicted Q value for the specific chosen action (that's why we multiply)
            # A much better strategy is to have the network multiply its outputs by a “mask” corresponding to the one-hot encoded action, which will set all its outputs to 0 except the one for the action we actually saw. We can then pass 0 as the target to for all unknown actions and our neural network should thus perform fine. When we want to predict for all actions, we can then simply pass a mask of all 1s.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)


            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

""" TRAINING
    Our algorithm:
        Initialize the weights
        Init the environment
        Initialize the decay rate (that will use to reduce epsilon)

        For episode to max_episode do
            Make new episode
            Set step to 0
            Observe the first state $s_0$

            While step < max_steps do:
                Increase decay_rate
                With $\epsilon$ select a random action $a_t$, otherwise select $a_t = \mathrm{argmax}_a Q(s_t,a)$
                Execute action $a_t$ in simulator and observe reward $r_{t+1}$ and new state $s_{t+1}$
                Store transition $
                Sample random mini-batch from $D$: $$
                Set $\hat{Q} = r$ if the episode ends at $+1$, otherwise set $\hat{Q} = r + \gamma \max_{a'}{Q(s', a')}$
                Make a gradient descent step with loss $(\hat{Q} - Q(s, a))^2$

            endfor

        endfor
    """
# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
model = DQNetwork(state_size, action_size, learning_rate, 'DQNetwork')
target_model = DQNetwork(state_size, action_size, learning_rate, 'Target_DQNetwork')

# Instantiate helper
helper = Helper()

# Instantiate memory & fill initial batch
memory = Memory(max_size = memory_size)

# Saver will help us to save our model
saver = tf.train.Saver()

if training == True:
    rewards_list = []

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75) # avoid crashing
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # On each game:
        for episode in range(total_episodes):
            # Initialize reward counter
            total_reward = 0

            # Observe the first state
            state = env.reset() # array containing pixels

            # Finished will tell if max_steps were reached or if the car is stuck
            finished = False

            for step in range(max_steps):
                if (helper.epsilon_greedy()):
                    # Make a random action
                    action = env.action_space.sample()

                else:
                    # Get action from Q-network
                    Qs = sess.run(model.output, feed_dict = {model.inputs_: state.reshape((1, *state.shape))})
                    action = np.argmax(Qs)
                    print(action)
                    if step % 7 == 0:
                        print(action)

                # Execute the action
                observation, reward, done, _ = env.step(action)

                # Add reward to counter
                total_reward += reward

                if done:
                    print(targets)
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)

                    # Keep track of how well each episode performed
                    rewards_list.append((episode, total_reward))

                    # Add info to memory
                    memory.add((state, possible_actions[action], reward, next_state, done))

                else:
                    # Get the next state
                    next_state = observation

                    # Add experience to memory
                    memory.add((state, possible_actions[action], reward, next_state, done))
                    state = next_state

                # LEARNING PART
                #    Take S, A (decided by network), R, S_, D
                #    Run S_ through the network, and get expected Q for each action
                #    Set Target Q to be equal to:
                #       Reward, if state is terminal (DQN will learn to predict if a certain action can be final)
                #       Reward + MaxQ for action selected by the DQN on t+1 (gamma considered)
                #    Train network with target Q

                # Obtain random mini-batch from memory
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                # Get target Qs for the batch
                targets = helper.getTargetQs(states, actions, rewards, next_states, dones, target_model, sess)

                # Calculate loss & optimize model
                loss, _ = sess.run([model.loss, model.optimizer],
                                    feed_dict={model.inputs_: states,
                                               model.target_Q: targets,
                                               model.actions_: actions})


                if done or finished:
                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Epsilon (exploration probability) P: {:.4f}'.format(helper.epsilon_probability))
                    print('________________________________________________')

                    break # end episode

            # Reduce epsilon for next game
            helper.reduceEpsilon()

            # Save model every 30 episodes
            if episode % 15 == 0:
                print("Model Saved")

        for episode in range(10):
            done = False
            step = 0
            totalScore = 0

            while not done:
                env.render()
                if step == 0:
                    observation = env.reset()
                    reward = 0

                else:
                    # Get action from network
                    Qs = sess.run(model.output, feed_dict = {model.inputs_: state.reshape((1, *state.shape))})
                    action = np.argmax(Qs)
                    observation, reward, done, _ = env.step(action)

                step += 1
                totalScore += reward

            # Reset step
            step = 0

            print("Score: ", totalScore)

    # Close env to avoid crashes
    env.close()
