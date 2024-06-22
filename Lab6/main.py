import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import sys
import random

def value_iteration(env, iterations=5e5, break_threshold=1e-3, gamma=0.9):
    idx = 0
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    V = np.zeros(num_states, dtype=np.float32)
    Q = np.zeros((num_states, num_actions), dtype=np.float32)

    for _ in range(int(iterations)):
        v_old = V.copy()
        for state in range(num_states):
            for action in range(num_actions):
                Q[state, action] = 0.0
                for transition in env.P[state][action]:
                    prob, next_state, r, _ = transition
                    Q[state, action] += prob * (r + gamma * v_old[next_state])
            V[state] = np.max(Q[state, :])
            idx += 1

        if np.max(np.abs(V-v_old)) < break_threshold:
            break

    return {s:V[s] for s in range(num_states)}

def eps_greedy_policy(env, Q, state, epsilon):
    if random.random() < epsilon:
        actions = list(env.P[state].keys()) 
        return random.choice(actions)
    else:
        return np.argmax(Q[state])

def generate_best_policy(env, Q):
    policy = {state: None for state in range(env.observation_space.n) }

    for state in range(env.observation_space.n):
        policy[state] = np.argmax(Q[state])

    return policy

def generate_values(Q):
    num_states, _ = Q.shape
    values = {state: None for state in range(num_states) }

    for state in range(num_states):
        values[state] = np.max(Q[state])

    return values

def compute_rmse(optimal_values, crt_values):
    num_states = len(optimal_values)

    rmse = 0
    for state in range(num_states):
        rmse += (optimal_values[state] - crt_values[state])**2

    return np.sqrt(rmse / num_states)

def q_learning(env, optimal_values, gamma, epsilon, alpha, iterations, env_name):

    if "Taxi" in env_name:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        Q = np.random.rand(env.observation_space.n, env.action_space.n) * 2 - 1
    rmses = []

    for _ in range(iterations):
        state = env.reset()
        state = state[0]
        done = False

        while not done:
            action = eps_greedy_policy(env, Q, state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state, action] = (1- alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))
            state = next_state

        crt_values = generate_values(Q)
        rmses.append(compute_rmse(optimal_values, crt_values))

    return rmses

def sarsa(env, optimal_values, gamma, epsilon, alpha, iterations, env_name):

    if "Taxi" in env_name:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        Q = np.random.rand(env.observation_space.n, env.action_space.n) * 2 - 1
    rmses = []

    for i in range(iterations):
        state = env.reset()
        state = state[0]
        done = False

        action = eps_greedy_policy(env, Q, state, epsilon)
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = eps_greedy_policy(env, Q, next_state, epsilon)
            done = terminated or truncated

            Q[state, action] = (1- alpha) * Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action])
            state = next_state
            action = next_action

        crt_values = generate_values(Q)
        rmses.append(compute_rmse(optimal_values, crt_values))

    return rmses

def nstep_sarsa(env, optimal_values, gamma, epsilon, alpha, n, iterations, env_name):
        
    if "Taxi" in env_name:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        Q = np.random.rand(env.observation_space.n, env.action_space.n) * 2 - 1
    G = np.zeros((env.observation_space.n, env.action_space.n))
    rmses = []

    for _ in range(iterations):
        state = env.reset()
        state = state[0]
        done = False

        T = sys.maxsize
        t = 0

        action = eps_greedy_policy(env, Q, state, epsilon)
        state_action_rewards = [(state, action, 0)]
        while t < T - 1:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                T = t + 1
            else:
                action = eps_greedy_policy(env, Q, next_state, epsilon)

            state_action_rewards.append((next_state, action, reward))

            tau = t - n + 1

            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += np.power(gamma, i - tau - 1) * state_action_rewards[i][2]

                if tau + n < T:
                    G += np.power(gamma, n) * Q[state_action_rewards[tau + n][0], state_action_rewards[tau + n][1]]

                state, action = state_action_rewards[tau][0], state_action_rewards[tau][1]
                Q[state, action] = (1- alpha) * Q[state, action] + alpha * G
            
            if tau == T - 1:
                break

            t += 1

        crt_values = generate_values(Q)
        rmses.append(compute_rmse(optimal_values, crt_values))

    return rmses


if __name__ == "__main__":
    random.seed(0)
    env_names = ["FrozenLake-v1", "Taxi-v3"]

    alphas = np.arange(0, 1.1, 0.2)
    ns = [2, 4, 6, 8]

    env_specs = []

    for env_name in env_names:
        if "Frozen" in env_name:
            env = gym.make(env_name, map_name="8x8")
        else:
            env = gym.make(env_name)
        optimal_v = value_iteration(env)

        qs_rmse_env = []
        sarsa_rmse_env = []
        nstep_rmse_env = [[] for _ in range(len(ns))]
        
        for alpha in alphas:
            print("alpha: ", alpha)
            qs_rmse = []
            sarsa_rmse = []
            nstep_rmse = [[] for _ in range(len(ns))]

            for iteration in range(20):
                print("iteration: ", iteration)

                qs_rmse.append(np.mean(q_learning(env, optimal_v, 0.9, 0.2, alpha, 200, env_name)))
                sarsa_rmse.append(np.mean(sarsa(env, optimal_v, 0.9, 0.2, alpha, 200, env_name)))

                for n in ns:
                    print("n: ", n)
                    nstep_rmse[ns.index(n)].append(np.mean(nstep_sarsa(env, optimal_v, 0.9, 0.2, alpha, n, 200, env_name)))

            qs_rmse_env.append(np.mean(qs_rmse))
            sarsa_rmse_env.append(np.mean(sarsa_rmse))
            
            for i in range(len(ns)):
                nstep_rmse_env[i].append(np.mean(nstep_rmse[i]))
     
        plt.plot(alphas, qs_rmse_env, label="Q-Learning")
        plt.plot(alphas, sarsa_rmse_env, label="SARSA")

        for i in range(len(ns)):
            plt.plot(alphas, nstep_rmse_env[i], label="{}-Step SARSA".format(ns[i]))

        plt.legend()
        plt.xlabel("Alpha")
        plt.ylabel("RMSE")
        plt.xticks(alphas)
        plt.title(env_name)
        #plt.show()
        plt.savefig(f"{env_name}_200.jpg")
        plt.close()
