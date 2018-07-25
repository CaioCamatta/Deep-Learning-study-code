import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from vizdoom import *        # Doom Environment
import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames
from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

# LOSS EXPLANATION : one network calculates Q(s,a), and subsequently (using the same approximator, like the same NN), the value of Q(s',a'), in which (s',a') are one time-step later than (s,a). Using a perfect Q-function, the difference between Q(s,a) and Q(s',a') should be exactly the direct reward received in going from s to s' by action a: Q(s,a)=Q(s′,a′)+r. Therefore, δ is computed as δ=Q(s′,a′)+r−Q(s,a), neglecting discount rate γ.

""" Here we create our environment """
def create_environment():
    game = DoomGame()

    # Load the correct configuration
    game.load_config("E:/Code/Anaconda/Learning/RL/DOOM_DQN/basic.cfg")

    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("E:/Code/Anaconda/Learning/RL/DOOM_DQN/basic.wad")

    game.init()

    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions
game, possible_actions = create_environment()

# Render the environment
game.new_episode()

""" HYPERPARAMETERS """
### MODEL HYPERPARAMETERS
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot
learning_rate =  0.0002      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 121        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64
rate_of_update = 0.001        # Rate in which the target network will copy the primary network's variables

# Exploration parameters for epsilon greedy strategy
epsilon_start = 0.9            # exploration probability at start
epsilon_stop = 0.1            # minimum exploration probability
decay_rate = 0.005            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.99

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size
memory_size = 50000

### PREPROCESSING HYPERPARAMETERS
stack_size = 4 # image stack size

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True


""" Convenient functions are placed inside the helper class for better organization """
class Helper:
    def __init__(self):
        # Initialize deque with zero-images one array for each image. 'maxlen=' is very important here
        self.stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Get list of trainable variables in both networks
        self.trainables = tf.trainable_variables()

        # Define operation that updates target network
        self.op_holder = self.updateTargetNetworkOp()

        # Start epsilon decay at 0
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
        # Greyscale frame already done in our vizdoom config
        # x = np.mean(frame,-1)

        # Crop the screen (remove the roof because it contains no information)
        cropped_frame = frame[30:-10,30:-30]

        # Normalize Pixel Values
        normalized_frame = cropped_frame/255.0

        # Resize
        preprocessed_frame = transform.resize(normalized_frame, [84,84])

        return preprocessed_frame

    def stack_frames(self, state):
        """ Stacks frames so that the AI can have a notion of movement """
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

""" Experience Replay """
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        self.fill_randomly(pretrain_length)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False) # False so we get unique instances

        return [self.buffer[i] for i in index]

    def fill_randomly(self, size):
        for i in range(size):
            if i == 0:
                # First we need a state
                state = game.get_state().screen_buffer
                state = helper.stack_frames(state)

            # Random action
            action = random.choice(possible_actions)

            # Get the rewards
            reward = game.make_action(action)

            # Look if the episode is finished
            done = game.is_episode_finished()


            if done:
                # We finished the episode
                next_state = np.zeros(state.shape)

                # Add experience to memory
                self.add((state, action, reward, next_state, done))

                # Start a new episode
                game.new_episode()
            else:
                # Get the next state
                next_state = game.get_state().screen_buffer
                next_state = helper.stack_frames(next_state)

                # Add experience to memory
                self.add((state, action, reward, next_state, done))

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
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                   name = 'batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [20, 20, 32]


            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                   name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [9, 9, 64]


            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                   name = 'batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            ## --> [3, 3, 128]

            """
            After CNN:
            Flatten
            Dense 512
            Dense action_space_size
            """
            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]


            self.fc = tf.layers.dense(inputs = self.flatten,
                                      units = 512,
                                      activation = tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")


            self.output = tf.layers.dense(inputs = self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = 3,
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

# TensorBoard
writer = tf.summary.FileWriter("E:/Code/Anaconda/Learning/RL/DOOM_DQN/tensorboard/dqn/1")
tf.summary.scalar("Loss", model.loss)
write_op = tf.summary.merge_all()

# Saver will help us to save our model
saver = tf.train.Saver()

if training == True:
    rewards_list = []

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75) # avoid crashing
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Init the game
        game.init()

        # On each game:
        for episode in range(total_episodes):
            # Make new episode
            game.new_episode()
            step = 0

            # Observe the first state
            frame = game.get_state().screen_buffer # array containing pixels
            state = helper.stack_frames(frame)

            for step in range(max_steps):
                if (helper.epsilon_greedy()):
                    # Make a random action
                    action = random.choice(possible_actions)

                else:
                    # Get action from Q-network
                    # Estimate the Qs values state
                    Qs = sess.run(model.output, feed_dict = {model.inputs_: state.reshape((1, *state.shape))})

                    # Take the biggest Q value (= the best action)
                    action = np.argmax(Qs)

                    # One hot encode action
                    action = possible_actions[int(action)]

                # Execute the action
                reward = game.make_action(action)

                # Look if the episode is finished
                done = game.is_episode_finished()

                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((84,84), dtype=np.int)
                    next_state = helper.stack_frames(next_state)

                    total_reward = game.get_total_reward()

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Epsilon (exploration probability) P: {:.4f}'.format(helper.epsilon_probability))

                    rewards_list.append((episode, total_reward))

                    memory.add((state, action, reward, next_state, done))

                else:
                    # Get the next state
                    next_state = game.get_state().screen_buffer
                    next_state = helper.stack_frames(next_state)

                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))
                    state = next_state

                # LEARNING PART
                #    Take S, A (decided by network), R, S_, D
                #    Run S_ through the network, and get expected Q for each action
                #    Set Target Q to be equal to:
                #       Reward, if state is terminal (DQN will learn to predict if a certain action can be final)
                #       Reward + MaxQ for action selected by the DQN on t+1 (gamma considered)
                #    Train network with target Q

                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch], ndmin=3)
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])
                dones = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state. Expected reward for each action
                target_Qs = sess.run(target_model.output, feed_dict = {target_model.inputs_: next_states})

                # Set Qhat = r if the episode ends at +1, otherwise set Qhat = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards[i])
                    else:
                        target = rewards[i] + gamma * np.max(target_Qs[i])
                        target_Qs_batch.append(target)


                targets = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([model.loss, model.optimizer],
                                    feed_dict={model.inputs_: states,
                                               model.target_Q: targets,
                                               model.actions_: actions})

                helper.updateTarget(sess) # update target network's variables with variables from primary network

                # Write TF/TBoard Summaries
                summary = sess.run(write_op, feed_dict={model.inputs_: states,
                                                   model.target_Q: targets,
                                                   model.actions_: actions})
                writer.add_summary(summary, episode)
                writer.flush()

                if done:
                    break # end episode

            # Reduce epsilon for next game
            helper.reduceEpsilon()

            # Save model every 30 episodes
            if episode % 30 == 0:
                save_path = saver.save(sess, "E:/Code/Anaconda/Learning/RL/DOOM_DQN/models/model.ckpt")
                print("Model Saved")

""" Watch the model play """
if(True):
    with tf.Session() as sess:
        game = DoomGame()
        helper = Helper()
        totalScore = 0

        # Load the correct configuration (test configuration)
        game.load_config("E:/Code/Anaconda/Learning/RL/DOOM_DQN/basic_test.cfg")

        # Load the correct scenario (in our case basic scenario)
        game.set_doom_scenario_path("E:/Code/Anaconda/Learning/RL/DOOM_DQN/basic.wad")

        # Load the model
        saver.restore(sess, "E:/Code/Anaconda/Learning/RL/DOOM_DQN/models/model.ckpt")
        game.init()
        for i in range(50):

            game.new_episode()
            while not game.is_episode_finished():
                frame = game.get_state().screen_buffer
                state = helper.stack_frames(frame)

                # Take the biggest Q value (= the best action)
                Qs = sess.run(model.output, feed_dict = {model.inputs_: state.reshape((1, *state.shape))})
                action = np.argmax(Qs)
                action = possible_actions[int(action)]
                game.make_action(action)
                score = game.get_total_reward()

            print("Score: ", score)
            totalScore += score

        print("TOTAL_SCORE", totalScore/100.0)
    game.close()
