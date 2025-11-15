# Kwangwoon University 
# Reinforcement Learning

# HW 2 - REINFORCE Algorithm Implementation
# Please fill in the missing parts marked with "TODO"
# You can use either CartPole-v1 or LunarLander-v3 environment from OpenAI Gym
# Make sure to have the required libraries installed:
# # pip install gymnasium torch tqdm numpy

# Submit your completed code file named reinforce.py
# and include a brief report of your results in report.pdf

import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
# You can switch between CartPole-v1 and LunarLander-v3

env_name = 'CartPole-v1'
#env_name = 'LunarLander-v3'
gamma = 0.9

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()

        # define the policy network (parameterized by theta)
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.reset_on_policy()
        self.train()  # set training mode

    def reset_on_policy(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        logits = self.model(x)
        return logits

    def act(self, state):
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state.astype(np.float32))  # to tensor
        
        # TODO: implement the following lines and fill the dots (....)
        # todo 영역 시작 ==========================================================
        
        """
        모델 로짓 확인
        로짓을 카테고리컬에 넣어 softmax 취한 것과 같은 효과
        softmax의 확률분포에서 액션 샘플링
        """
        logit = self.forward(x)  # forward pass
        pd = Categorical(logits=logit)  # probability distribution
        action = pd.sample()
        
        # you don't need to calculate the gradient for log_prob here
        # instead, the torch will take care of it during backpropagation step
        """손실용 ln(pi(a|s;\theta)) 저장"""
        selected_action_log_prob = pd.log_prob(action)
        self.log_probs.append(selected_action_log_prob)
        return action.item()
        # todo 영역 끝 ==========================================================


def train(pi, optimizer):
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    returns = np.empty(T, dtype=np.float32)  # the returns
    G = 0.0
    # compute the returns efficiently
    for t in reversed(range(T)):
        # todo 영역 시작 ==========================================================
        """누적합 : 할인율 * 관측보상"""
        G = gamma * G + pi.rewards[t]
        # todo 영역 끝 ==========================================================
        returns[t] = G

    returns = torch.tensor(returns)
    log_probs = torch.stack(pi.log_probs)
    # todo 영역 시작 ==========================================================
    """
    정책경사정리에 따라
    성능의 경사 = E[sum_{t} G_t * nabla_{theta} ln(pi(a|s;theta))]
    따라서 딥러닝 프레임워크를 통해 최적화 할 시 위 경사와 미분이 같은 손실을 주면 된다
    위 정리에 의해 나블라를 없애면 성능함수가 나오고 딥러닝 프레임워크는 경사하강을 수행하므로
    -(성능) = -E[sum_{t} G_t * ln(pi(a|s;theta))] 을 손실로 삼으면 된다
    특별히 이 프로그램에서는 에피소드 한 번에 업데이트 한 번 수행(|batch|=1인 sgd)하므로
    손실 = - sum_{t} G_t * ln(pi(a|s;theta))
    """
    loss = -(returns * log_probs)  # TODO: gradient term; Negative for maximizing
    # todo 영역 끝 ==========================================================
    loss = torch.sum(loss)

    # optimization step (you do not need to change this part)
    optimizer.zero_grad()
    loss.backward()  # backpropagate, compute gradients
    optimizer.step()  # gradient-ascent, update the weights
    
    return loss


def train_cartpole():
    env = gym.make(env_name)
    
    in_dim = env.observation_space.shape[0]  
    out_dim = env.action_space.n 

    pi = Pi(in_dim, out_dim)  # policy pi_theta for REINFORCE
    optimizer = optim.Adam(pi.parameters(), lr=0.005)

    best_pi = None
    best_reward = -float('inf')

    tq = tqdm.tqdm(range(100))
    for epi in tq:
        state, _ = env.reset()

        # new episode
        for t in range(2000):  # cartpole max timestep is 2000

            # step
            action = pi.act(state)
            state, reward, done, truncated, info = env.step(action)

            pi.rewards.append(reward)

            env.render()
            if done:
                break

        # optimization (similar to the gradient-ascent step)

        loss = train(pi, optimizer)  # train per episode
        total_reward = sum(pi.rewards)

        pi.reset_on_policy()  # onpolicy: clear memory after training

        tq.set_description(f'Episode {epi:6d}, loss: {loss.item():8.2f}, total_reward: {total_reward:6.2f}')
        if total_reward > best_reward:
            best_reward = total_reward
            best_pi = Pi(in_dim, out_dim)
            best_pi.load_state_dict(pi.state_dict())
            print(f' New best policy. Episode {epi}, total_reward: {total_reward}')

    env.close()
    return best_pi


def display_policy(pi):
    env = gym.make(env_name, render_mode='human')
    state, _ = env.reset()
    done = False
    reward_sum = 0
    trial, mean_reward = 10, 0
    for epi in range(trial):
        state, _ = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action = pi.act(state)
            state, reward, done, truncated, info = env.step(action)
            reward_sum += reward
        mean_reward += reward_sum
        print(f'Total reward with trained policy: {reward_sum}')
    
    mean_reward /= trial
    print(f'Mean reward with {trial} trial: {mean_reward}')

    env.close()


def load_policy(filename='pi.pth'):
    env = gym.make(env_name)
    in_dim = env.observation_space.shape[0] 
    out_dim = env.action_space.n  
    pi = Pi(in_dim, out_dim)  
    pi.load_state_dict(torch.load(filename))
    print(f'Policy loaded from {filename}')
    return pi


def save_policy(pi, filename='pi.pth'):
    torch.save(pi.state_dict(), filename)
    print(f'Policy saved to {filename}')


def main():
    # Choose whether to train a new policy or load an existing one
    use_train = True

    if use_train:
        pi = train_cartpole()
        save_policy(pi, f'{env_name}_pi.pth')
    else:
        pi = load_policy(f'{env_name}_pi.pth')

    display_policy(pi)


if __name__ == '__main__':
    main()
