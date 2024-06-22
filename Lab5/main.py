import pickle
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random

from argparse import ArgumentParser
from tqdm import tqdm
from itertools import product, combinations

def eps_greedy_policy(env, Q, state, epsilon):
    if random.random() < epsilon:
        actions = list(range(env.action_space.n)) 
        return random.choice(actions)
    else:
        return np.argmax(Q[state])

def generate_best_policy(env, Q):
    policy = {state: None for state in range(env.observation_space.n)}

    for state in range(env.observation_space.n):
        policy[state] = np.argmax(Q[state])

    return policy

def simulate_policy(env, Q, iterations):
    epoch_rewards = []

    policy = generate_best_policy(env, Q)

    for i in range(iterations):
        state = env.reset()
        state = state[0]
        done = False

        total_reward = 0
        while not done:
            action = policy[state]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state

        epoch_rewards.append(total_reward)

    return np.mean(epoch_rewards)
    
def q_learning(env, gamma, epsilon, alpha, iterations, _evaluation_step = 100):
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    train_rewards = []
    crt_rewards = []
    test_rewards = []

    for i in range(iterations):
        state = env.reset()
        state = state[0]
        done = False
        total_reward = 0

        while not done:
            action = eps_greedy_policy(env, Q, state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            Q[state, action] = (1- alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))
            state = next_state

        if i % _evaluation_step == 0 and i != 0:
            test_rewards.append(simulate_policy(env, Q, iterations=_evaluation_step))
            train_rewards.append(np.mean(crt_rewards[-_evaluation_step:]))

        crt_rewards.append(total_reward)

    return train_rewards, test_rewards

def sarsa(env, gamma, epsilon, alpha, iterations, eval_iter = 100):
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    train_rewards = []
    crt_rewards = []
    test_rewards = []

    for i in range(iterations):
        state = env.reset()
        state = state[0]
        done = False
        total_reward = 0

        action = eps_greedy_policy(env, Q, state, epsilon)
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = eps_greedy_policy(env, Q, next_state, epsilon)
            done = terminated or truncated

            total_reward += reward
            Q[state, action] = (1- alpha) * Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action])
            state = next_state
            action = next_action

        if i % eval_iter == 0 and i != 0:
            test_rewards.append(simulate_policy(env, Q, iterations=eval_iter))
            train_rewards.append(np.mean(crt_rewards[-eval_iter:]))
            
        crt_rewards.append(total_reward)

    return train_rewards, test_rewards

def plot_for_env(env_name, q_tr, q_te, sarsa_tr, sarsa_te):
    plt.figure()
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.suptitle(env_name)
    ax[0].plot(list(range(len(q_tr))), q_tr, label="Q-Learning train")
    ax[0].plot(list(range(len(sarsa_tr))), sarsa_tr, label="SARSA train")
    ax[0].legend(loc="upper right")
    ax[1].plot(list(range(len(q_te))), q_te, label="Q-Learning test")
    ax[1].plot(list(range(len(sarsa_te))), sarsa_te, label="SARSA test")
    ax[1].legend(loc="upper right")
    plt.savefig(f"{env_name}.png")
    plt.close()

def plot_params(param_names, env_name, res):
    for k in res:
        param1, param2, (val1, val2) = k
        fig_title = f"{env_name}_{param1}={val1}_{param2}={val2}"
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(19.2, 10.8))
        fig.suptitle(fig_title)
    
        other = list(set(param_names)-set([param1, param2]))[0]
        for other_val, data in res[k].items():
            q_tr, q_te, sarsa_tr, sarsa_te = data
            ax[0].plot(list(range(len(q_tr))), q_tr, label=f"QL train ({other}={other_val})")
            ax[0].plot(list(range(len(sarsa_tr))), sarsa_tr, label=f"SARSA train ({other}={other_val})")
            ax[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
            ax[1].plot(list(range(len(q_te))), q_te, label=f"QL test ({other}={other_val})")
            ax[1].plot(list(range(len(sarsa_te))), sarsa_te, label=f"SARSA test ({other}={other_val})")
            ax[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        plt.savefig(f"{fig_title}.png")
        plt.close()

def parse_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--plot_from_cache",
        required=False,
        action="store_true"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(0)
    names = ["Q-Learning", "SARSA"]
    env_names = ["Taxi-v3", "FrozenLake-v1"]

    # gamma = 0.9
    # epsilon = 0.5
    # alpha = 0.5
    # for env_name in env_names:
    #     env = gym.make(env_name)
    #     for name in names:
    #         if name == "Q-Learning":
    #             q_tr, q_te = q_learning(env, gamma, epsilon, alpha, 10000, 50)
    #         else:
    #             sarsa_tr, sarsa_te = sarsa(env, gamma, epsilon, alpha, 10000, 50)

    #     plot_for_env(env_name, q_tr, q_te, sarsa_tr, sarsa_te)


    gammas = [0.5, 0.9]
    epsilons = [0.1, 0.5, 0.8]
    alphas = [0.1, 0.5, 0.9]
    combs = list(product(gammas, epsilons, alphas))
    param_names = ["gamma", "epsilon", "alpha"]

    for env_name in env_names:
        if not args.plot_from_cache:
            res = {}
            env = gym.make(env_name)
            for elem in tqdm(combs):
                gamma, epsilon, alpha = elem
                for name in names:
                    if name == "Q-Learning":
                        q_tr, q_te = q_learning(env, gamma, epsilon, alpha, 1000, 50)
                    else:
                        sarsa_tr, sarsa_te = sarsa(env, gamma, epsilon, alpha, 1000, 50)
                
                p = list(zip(param_names, elem))
                z = combinations(p, 2)
                for comb in z:
                    (name1, val1), (name2, val2) = comb
                    other, other_val = list(set(p)-set(comb))[0]
                    k = (name1, name2, (val1, val2))
                    if k not in res:
                        res[k] = {}
                    res[k][other_val] = [q_tr, q_te, sarsa_tr, sarsa_te]
        else:
            with open(f"{env_name}.pkl", "rb") as fin:
                res = pickle.load(fin)

        plot_params(param_names, env_name, res)

        with open(f"{env_name}.pkl", "wb") as fout:
            pickle.dump(res, fout)

if __name__ == "__main__":
    main()