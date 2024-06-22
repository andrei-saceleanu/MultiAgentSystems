import os
import numpy as np
import matplotlib.pyplot as plt

from algorithms import *
from gridworld import GridWorld

from itertools import product

def plot_for_env(env_name, q_tr, q_te, sarsa_tr, sarsa_te, dq_tr, dq_te, num_actions, filename_data):
    plt.figure()
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.suptitle(env_name)
    ax[0].plot(list(range(len(q_tr))), q_tr, label="Q-Learning train")
    ax[0].plot(list(range(len(sarsa_tr))), sarsa_tr, label="SARSA train")
    ax[0].plot(list(range(len(dq_tr))), dq_tr, label="Double Q-Learning train")
    ax[0].legend(loc="lower right")
    ax[1].plot(list(range(len(q_te))), q_te, label="Q-Learning test")
    ax[1].plot(list(range(len(sarsa_te))), sarsa_te, label="SARSA test")
    ax[1].plot(list(range(len(dq_te))), dq_te, label="Double Q-Learning test")
    ax[1].legend(loc="lower right")
    image_folder = filename_data["dir"]
    x = {k:v for k,v in filename_data.items() if k!="dir"}
    plt.savefig(os.path.join(image_folder, f"{env_name}_{num_actions}_{'_'.join([k+'='+str(v) for k,v in x.items()])}.png"))
    plt.close()

def plot_one(env_name, q_tr, q_te, num_actions, filename_data):
    plt.figure()
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.suptitle(env_name)
    ax[0].plot(list(range(len(q_tr))), q_tr, label="Q-Learning train")
    ax[0].legend(loc="lower right")
    ax[1].plot(list(range(len(q_te))), q_te, label="Q-Learning test")
    ax[1].legend(loc="lower right")
    image_folder = filename_data["dir"]
    x = {k:v for k,v in filename_data.items() if k!="dir"}
    plt.savefig(os.path.join(image_folder, f"{env_name}_{num_actions}_{'_'.join([k+'='+str(v) for k,v in x.items()])}.png"))
    plt.close()
    
def plot_lens(cfg, qlens, slens, dqlens, num_actions, filename_data):
    plt.figure()
    plt.plot(list(range(len(qlens))), qlens, label="Q lengths")
    plt.plot(list(range(len(slens))), slens, label="SARSA lengths")
    plt.plot(list(range(len(dqlens))), dqlens, label="Double Q lengths")
    plt.legend()
    image_folder = filename_data["dir"]
    x = {k:v for k,v in filename_data.items() if k!="dir"}
    plt.savefig(os.path.join(image_folder, f"ep_lens_{cfg['world_type']}_{num_actions}_{'_'.join([k+'='+str(v) for k,v in x.items()])}.png"))
    plt.close()

def plot_lens_one(cfg, qlens, num_actions, filename_data):
    plt.figure()
    plt.plot(list(range(len(qlens))), qlens, label="Q lengths")
    plt.legend()
    image_folder = filename_data["dir"]
    x = {k:v for k,v in filename_data.items() if k!="dir"}
    plt.savefig(os.path.join(image_folder, f"task3_{cfg['approach']}_ep_lens_{cfg['world_type']}_{num_actions}_{'_'.join([k+'='+str(v) for k,v in x.items()])}.png"))
    plt.close()
    
def getPolicies(qQ, sQ, dqQ, cfg, world, num_actions, filename_data):
    mapqQ = np.argmax(qQ, axis=1)
    mapqQ = mapqQ.reshape(cfg["world_size"][0], cfg["world_size"][1]) # they are reversed
    mapsQ = np.argmax(sQ, axis=1)
    mapsQ = mapsQ.reshape(cfg["world_size"][0], cfg["world_size"][1]) # they are reversed
    mapdqQ = np.argmax(dqQ, axis=1)
    mapdqQ = mapdqQ.reshape(cfg["world_size"][0], cfg["world_size"][1]) # they are reversed
    
    if num_actions == 4:
        actions_mapping = {
            0: '^',
            1: '->',
            2: 'v',
            3: '<-'
        }
    elif num_actions == 8:
        actions_mapping = {
            0: '^',
            1: 'NE',
            2: '->',
            3: 'SE',
            4: 'v',
            5: 'SW',
            6: '<-',
            7: 'NW'
        }
    cmap = plt.get_cmap('tab10', len(actions_mapping))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, matrix in zip(axs[:-1], [mapqQ, mapsQ, mapdqQ]):
        ax.imshow(matrix, cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(axs[2].imshow(mapqQ, cmap=cmap, interpolation='nearest'), ax=axs[-1], fraction=0.03, pad=0.04, ticks=np.arange(len(actions_mapping)))
    cbar.ax.set_yticklabels([actions_mapping[i] for i in range(len(actions_mapping))])

    axs[0].set_title('Q Learning')
    axs[1].set_title('Sarsa')
    axs[2].set_title('Double Q Learning')
    for ax in axs:
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
    plt.tight_layout()
    image_folder = filename_data["dir"]
    x = {k:v for k,v in filename_data.items() if k!="dir"}
    plt.savefig(os.path.join(image_folder, f"policies_{world}_{num_actions}_{'_'.join([k+'='+str(v) for k,v in x.items()])}.png"))
    plt.close()
    
def main():

    cfg = {
        "num_actions": 4,
        "world_size": [7, 10],
        "world_type": "A",
        "start_state_idx": 0,
        "task3": False,
        "approach": 1
    }
    env = GridWorld(cfg=cfg)

    random.seed(0)
    names = ["Q-Learning", "SARSA", "Double Q-Learning"]

    gamma = 0.999
    num_steps = 30_000
    eval_iter = 100
    eps_list = [0.1, 0.3, 0.5]
    alpha_list = [0.45, 0.75, 0.9]
    os.makedirs("images", exist_ok=True)
    if cfg['task3']:
        num_steps = 1_000_000 # for task3
        eval_iter = 100# for task3
        epsilon = eps_list[0]
        alpha = alpha_list[2]
        if cfg['approach'] == 1:
            q_tr, q_te, qlens, qQ = q_learning_3_ag_old(env, 0.9,  epsilon, alpha, num_steps, eval_iter)
        else:
            q_tr, q_te, qlens, qQ = q_learning_3_ag(env, 0.9,  epsilon, alpha, num_steps, eval_iter)
        
        filename_data = {"dir": "images", "start": cfg["start_state_idx"], "eps": epsilon, "alpha": alpha, "approach": cfg['approach']}
        plot_one(f"GridWorld_{cfg['world_type']}_{cfg['approach']}" + ("" if not cfg['task3'] else "_task3"), q_tr, q_te, cfg["num_actions"], filename_data)
        plot_lens_one(cfg, qlens, cfg["num_actions"], filename_data)
        print(q_tr[-1])
    else:
        for epsilon, alpha in product(eps_list, alpha_list):
            print(f"EPS={epsilon}; ALPHA={alpha}")
            for name in names:
                if name == "Q-Learning":
                    q_tr, q_te, qlens, qQ = q_learning(env, gamma, epsilon, alpha, num_steps, eval_iter)
                elif name == "SARSA":
                    sarsa_tr, sarsa_te, slens, sQ = sarsa(env, gamma, epsilon, alpha, num_steps, eval_iter)
                elif name == "Double Q-Learning":
                    dq_tr, dq_te, dqlens, dqQ = double_q_learning(env, gamma, epsilon, alpha, num_steps, eval_iter)

            filename_data = {"dir": "images", "start": cfg["start_state_idx"], "eps": epsilon, "alpha": alpha}
            getPolicies(qQ, sQ, dqQ, cfg, cfg['world_type'], cfg["num_actions"], filename_data)
            plot_for_env(f"GridWorld_{cfg['world_type']}", q_tr, q_te, sarsa_tr, sarsa_te, dq_tr, dq_te, cfg["num_actions"], filename_data)
            plot_lens(cfg, qlens, slens, dqlens, cfg["num_actions"], filename_data)

if __name__=="__main__":
    main()