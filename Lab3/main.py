import gym
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():

    ENV_NAME = "Taxi-v3" #"FrozenLake-v1"#,"FrozenLake8x8-v1"
    MAX_ITER = int(5e5)
    ALGOS = ["VI", "GS", "PI"]
    # ALGOS = ["PS"]
    EPS = 1e-3
    gamma = 0.9
    env = gym.make(ENV_NAME)
   
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    if "VI" in ALGOS:
        vs, V = value_iteration(MAX_ITER, EPS, gamma, env, num_states, num_actions)
        norm_diffs = [np.linalg.norm(elem-V) for elem in vs]
        x = list(range(len(norm_diffs)))
        plt.plot(x, norm_diffs, label="VI")
    if "GS" in ALGOS:
        vs, V = gauss_seidel(MAX_ITER, EPS, gamma, env, num_states, num_actions)
        norm_diffs = [np.linalg.norm(elem-V) for elem in vs]
        x = list(range(len(norm_diffs)))
        plt.plot(x, norm_diffs, label="GS")
    if "PI" in ALGOS:
        vs, V = policy_iteration(MAX_ITER, EPS, gamma, env, num_states, num_actions)
        norm_diffs = [np.linalg.norm(elem-V) for elem in vs]
        x = list(range(len(norm_diffs)))
        plt.plot(x, norm_diffs, label="PI")
    if "PS" in ALGOS:
        idx = 0
        vs = []
        V = np.zeros(num_states, dtype=np.float32)
        Q = np.zeros((num_states, num_actions), dtype=np.float32)
        H = np.zeros(num_states, dtype=np.float32)
        dependent_states=[]
        for _ in range(num_states):
            dependent_states.append([])
        for s in range(num_states):
           for a in range(num_actions):
              for temp in env.P[s][a]:
                  prob, nextstate, _, _ = temp
                  already_states = [st for st,_ in dependent_states[nextstate]]
                  if s not in already_states:
                    dependent_states[nextstate].append((s, prob))
    
        prev_state = None
        for _ in range(MAX_ITER):
            state = np.argsort(H)[::-1]
            if prev_state is None:
                state = state[0]
            else:
                state = state[1]
            prev_state = state

            v_old = V.copy()
            for action in range(num_actions):
                Q[state, action] = 0.0
                for transition in env.P[state][action]:
                    prob, next_state, r, _ = transition
                    Q[state, action] += prob * (r + gamma * v_old[next_state])
            V[state] = np.max(Q[state, :])
            idx += 1    
            vs.append(V.copy())
            H[state] = 0.0
            delta = abs(V[state]-v_old[state])
            
            for e, prob in dependent_states[state]:
                print(e, prob, delta)
                H[e] = max(H[e], delta * prob)
            print(H)
            print("===========")
            
            if np.max(np.abs(V-v_old)) < EPS:
                break
    
        norm_diffs = [np.linalg.norm(elem-V) for elem in vs]
        x = list(range(len(norm_diffs)))
        plt.plot(x, norm_diffs, label="PS")
    
    plt.title(f"{ENV_NAME} - MAX_ITER={MAX_ITER},EPS={EPS},gamma={gamma}")
    plt.legend(loc="upper right")
    plt.xlabel("#Iterations")
    plt.ylabel("||V-V*||")
    plt.show()

def policy_iteration(MAX_ITER, EPS, gamma, env, num_states, num_actions):
    running_list = []
    for _ in range(5):
        idx = 0
        vs = []
        pi = np.random.choice(num_actions,num_states)
        V = np.zeros(num_states, dtype=np.float32)
        Q = np.zeros((num_states, num_actions), dtype=np.float32)
    
        while True:
            for _ in range(MAX_ITER):
                v_old = V.copy()
                for state in range(num_states):
                    # V[state] = 0.0
                    for transition in env.P[state][pi[state]]:
                        prob, next_state, r, _ = transition
                        V[state] += prob * (r + gamma * v_old[next_state])
                    vs.append(V.copy())
                    idx += 1
                    
                if np.max(np.abs(V-v_old)) < EPS:
                    break

            stable=True
            for state in range(num_states):
                for action in range(num_actions):
                    Q[state, action] = 0.0
                    for transition in env.P[state][action]:
                        prob, next_state, r, _ = transition
                        Q[state, action] += prob * (r + gamma * V[next_state])
            new_pi = np.argmax(Q, axis=1)
            for state in range(num_states):
                act_old = pi[state]
                if act_old != new_pi[state]:
                    pi[state] = new_pi[state]
                    stable=False
            if stable == True:
                break 
        running_list.append(idx)
    print(np.mean(running_list))
    print(running_list)
    return vs, V


def policy_iteration2(MAX_ITER, EPS, gamma, env, num_states, num_actions):
    running_list = []
    for _ in range(5):
        idx = 0
        vs = []
        pi = np.random.choice(num_actions,num_states)
        V = np.zeros(num_states, dtype=np.float32)
        Q = np.zeros((num_states, num_actions), dtype=np.float32)
    
        while True:
            for _ in range(MAX_ITER):
                v_old = V.copy()
                for state in range(num_states):
                    V[state] = 0.0
                    for transition in env.P[state][pi[state]]:
                        prob, next_state, r, _ = transition
                        V[state] += prob * (r + gamma * v_old[next_state])
                    vs.append(V.copy())
                    idx += 1
                    
                if np.max(np.abs(V-v_old)) < EPS:
                    break

            stable=True
            for state in range(num_states):
                for action in range(num_actions):
                    Q[state, action] = 0.0
                    for transition in env.P[state][action]:
                        prob, next_state, r, _ = transition
                        Q[state, action] += prob * (r + gamma * V[next_state])
            new_pi = np.argmax(Q, axis=1)
            for state in range(num_states):
                act_old = pi[state]
                if act_old != new_pi[state]:
                    pi[state] = new_pi[state]
                    stable=False
            if stable == True:
                break 
        running_list.append(idx)
    print(np.mean(running_list))
    print(running_list)
    return vs, V




def gauss_seidel(MAX_ITER, EPS, gamma, env, num_states, num_actions):
    idx = 0
    vs = []
    V = np.zeros(num_states, dtype=np.float32)
    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    
    for _ in range(MAX_ITER):
        v_old = V.copy()
        for state in range(num_states):
            for action in range(num_actions):
                Q[state, action] = 0.0
                for transition in env.P[state][action]:
                    prob, next_state, r, _ = transition
                    Q[state, action] += prob * (r + gamma * V[next_state])
            V[state] = np.max(Q[state, :])
            idx += 1    
            vs.append(V.copy())

        if np.max(np.abs(V-v_old)) < EPS:
            break

    return vs, V

def value_iteration(MAX_ITER, EPS, gamma, env, num_states, num_actions):
    idx = 0
    vs = []
    V = np.zeros(num_states, dtype=np.float32)
    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    
    for _ in range(MAX_ITER):
        v_old = V.copy()
        for state in range(num_states):
            for action in range(num_actions):
                Q[state, action] = 0.0
                for transition in env.P[state][action]:
                    prob, next_state, r, _ = transition
                    Q[state, action] += prob * (r + gamma * v_old[next_state])
            V[state] = np.max(Q[state, :])
            idx += 1    
            vs.append(V.copy())

        if np.max(np.abs(V-v_old)) < EPS:
            break

    return vs, V


    


if __name__=="__main__":
    main()