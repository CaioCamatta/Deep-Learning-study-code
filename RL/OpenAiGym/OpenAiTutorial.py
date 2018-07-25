import gym
import random
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
import uuid

env = gym.make('CartPole-v0')
env.reset()

# PARAMETERS
LR = 1e-3
goal_steps = 500
score_requirement = 50 # only learn from games that achieve at least 50 score/steps.
initial_games = 20000

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
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            if done:
                break

        # Save game data to training data if the score is acceptable
        if score >= score_requirement:
            accepted_scores.append(score)

            for data in game_memory:
                # ONE-HOT ENCODE. Just a good practice since most games have more than 2 actions
                if data[1] == 1:
                    output = [0,1]
                if data[1] == 0:
                    output = [1,0]

                training_data.append([data[0],output])

        env.reset()
        scores.append(score)

    # Save data
    #np.save('saved.npy', np.array(training_data))

    # Print results
    print('Average accepted scores: ', mean(accepted_scores))
    print('Median accepted scores: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data



def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_observation = []

    env.reset()

    for _ in range(goal_steps):
        env.render()

        if len(prev_observation) == 0: # if there's nothing
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_observation.reshape(-1, len(prev_observation), 1))[0])

        # Save our choices to check if the network is working fine
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_observation = new_observation

        # retrain
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break

    scores.append(score)
print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(score_requirement)
