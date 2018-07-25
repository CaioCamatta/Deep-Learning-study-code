# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/

### The Idea ###
# Environment
#    run()           # runs one episode

# Agent
#    act(s)          # decides what action to take in state s
#    observe(sample) # adds sample (s, a, r, s_) to memory
#    replay()        # replays memories and improves

# Brain
#    predict(s)      # predicts the Q function values in state s
#    train(batch)    # performs supervised training step with batch

# Memory
#    add(sample)     # adds sample to memory
#    sample(n)       # returns random batch of n

import random, numpy, math, gym

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        """ Creates the model """
        set_session(tf.Session(config=config))
        model = Sequential()

        model.add(Dense(output_dim=64, activation='relu', input_dim=stateCnt))
        model.add(Dense(output_dim=actionCnt, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        """ Trains the model using a provided batch """
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        """ Returns a prediction. """
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCnt)).flatten()

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        """ Adds and entry to the memory ( s, a, r, s_ )"""
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        """ Returns random ( s, a, r, s_ ) tuples."""
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        """ Choose wheter to act randomly of according to a prediction """
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        """ Adds sample to the memory and updates epsilon """
        self.memory.add(sample)

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        # dummy replacement for when the state is final (Q(s,a) -> r)
        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ]) # state on t
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ]) # state on t+1 ([0,0,0,0] if final)

        p = self.brain.predict(states) # predictions for the starting states for each sample. Used as a default target in the learning.
        p_ = self.brain.predict(states_) # predictions for final states. Used in maxQ(s_,a).


        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]

            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i] # prediction for s

            if s_ is None:
                t[a] = r # update Q (t[1 or 2]) to be equal to reward if the state is final.
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i]) # If the state isn't final, update normally


            x[i] = s
            y[i] = t

        self.brain.train(x, y)


#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent):
        s = self.env.reset()
        R = 0

        while True:
            #self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) ) # save observartion & update epsilon
            agent.replay()

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)

#-------------------- MAIN ----------------------------
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

for _ in range(2):
    env.run(agent)
agent.brain.model.save("cartpole-basic.h5")
