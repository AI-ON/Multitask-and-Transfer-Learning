import subprocess
import os.path
import time

import numpy as np

import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as cg

import gym
from gym.envs.classic_control import rendering

from models.predictive_autoencoder import PredictorAgent

ROUNDS = 200000
BACKPROP_ROUNDS = 10
SAVE_DIR = 'saves/'
DIAGRAM_DIR = 'diagrams/'
SAVE_INTERVAL_SECONDS = 30

def process_image(raw_image):
    floated = raw_image.astype('float32') / 255.0
    transposed = F.transpose(floated)
    expanded = F.expand_dims(transposed, 0) # Make a "batch size" of 1
    return expanded

def to_image_format(arr):
    return (arr * 255).astype('int8')


def main(game_name, model_name):
    env = gym.make(game_name)
    agent = PredictorAgent(SAVE_DIR, name=model_name)
    viewer2 = rendering.SimpleImageViewer()

    last_save_time = agent.save()
    reset = True

    for i in range(ROUNDS):
        if reset:
            reset_obs = process_image(env.reset())
            agent.initialize_state(reset_obs)
            action = agent(reset_obs, reward=0)
            if action >= env.action_space.n:
                action = 0  # no-op
            env.render()
        raw_obs, reward, reset, _data = env.step(action)
        obs = process_image(raw_obs)
        action = agent(obs, reward)
        if action >= env.action_space.n:
            action = 0  # no-op

        # Render things prediction
        env.render()
        viewer2.imshow(to_image_format(agent.predicted_image.data))

        if time.time() - last_save_time > SAVE_INTERVAL_SECONDS:
            last_save_time = agent.save()
            print('Saved the', agent.name, 'agent')


MODELS = {
    'predictive_autoencoder': PredictorAgent
}


if __name__ == '__main__':
    main('Pong-v0', 'predictive_autoencoder')
