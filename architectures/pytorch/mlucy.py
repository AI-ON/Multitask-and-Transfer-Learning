import gym
import torch
import torch.nn as nn
import time
import random
import math
from torch.autograd import Variable

def from_gym(x):
    return torch.from_numpy(x).transpose(0, 2).unsqueeze(0).float()

class BreakoutActorCritic(nn.Module):
    def __init__(self, input_shape, output_size):
        super().__init__()
        in_channels = input_shape[1]
        self.visual_parser = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU())
        input_sz = self.visual_parser(Variable(torch.zeros(input_shape))).view(-1).data.shape[0]
        self.common = nn.Sequential(
            nn.Linear(input_sz, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(256, output_size), nn.Softmax())
        self.value = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, state0, state1):
        state = Variable(torch.cat((state0, state1), 1)).cuda()
        c = self.common(self.visual_parser(state).view(-1).unsqueeze(0))
        return self.action(c), self.value(c)

class BreakoutPlayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.breakout = gym.make('Breakout-v4')
        self.setup()
        self.model = model
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-7)
        self.value_criterion = nn.MSELoss()
        self.gamma = 0.99
        self.last_action = None

    def setup(self):
        self.states = [from_gym(self.breakout.reset())]
        self.states.append(from_gym(self.breakout.step(0)[0]))
        self.last_action = None

    def step(self, should_render):
        action_probs, value = self.model.forward(*self.states)
        action = action_probs.multinomial()

        if self.last_action is not None:
            self.opt.zero_grad()
            ideal_value = self.last_reward + self.gamma*Variable(value.data)
            value_diff = ideal_value - self.last_value
            self.last_action.reinforce(value_diff.data)
            self.last_action.backward(retain_graph=True)
            value_loss = self.value_criterion(self.last_value, ideal_value)
            value_loss.backward()
            self.opt.step()

        self.last_action = action
        self.last_value = value

        to_take = action.data[0][0]

        if should_render:
            print('Action: %s' % ['.', 'F', 'R', 'L'][to_take])
            print('Value: %s' % value.data[0][0])
            d = action_probs.data[0]
            print('.=%.02f F=%.02f R=%.02f L=%.02f' % (d[0], d[1], d[2], d[3]))
            self.breakout.render()

        state, self.last_reward, done, _ = self.breakout.step(to_take)
        if done:
            self.opt.zero_grad()
            ideal_value = Variable(torch.Tensor(((0,),)).cuda())
            value_loss = self.value_criterion(self.last_value, ideal_value)
            value_loss.backward(retain_graph=True)
            value_diff = ideal_value - self.last_value
            self.last_action.reinforce(value_diff.data)
            self.last_action.backward()
            self.opt.step()

            self.setup()
        else:
            self.states = [self.states[1], from_gym(state)]

fake_breakout = gym.make('BreakoutNoFrameskip-v4')
state_shape = list(from_gym(fake_breakout.observation_space.low).shape)
state_shape[1] *= 2 # we have one frame of history
model = BreakoutActorCritic(state_shape, fake_breakout.action_space.n)
bps = [BreakoutPlayer(model).cuda() for _ in range(16)]
bps[0].breakout.render()
while True:
    for i in range(8):
        bps[i].step(True)

