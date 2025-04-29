import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from survival_based_agent import SurvivalBasedAgent
from survival_based_wrapper import SurvivalBasedPacmanWrapper
from constants import ACTION_DIM, UP, DOWN, LEFT, RIGHT

# --- Utility Functions ---

def compute_mean_ci(data, confidence=0.95):
    data = np.array(data)
    mean_val = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean_val, ci

def evaluate(agent, episodes=10):
    env = SurvivalBasedPacmanWrapper()
    scores, steps_list, survival_ratios = [], [], []
    completion_rate = 0
    for episode in range(episodes):
        state = env.reset()
        total_reward, steps = 0, 0
        agent.survival_mode_count, agent.explore_mode_count = 0, 0
        done = False
        while not done and steps < 5000:
            action, survival_mode, _ = agent.choose_action(state)
            next_state, reward, done = env.take_action(action)
            state = next_state
            total_reward += reward
            steps += 1
            if steps % 3 == 0:
                time.sleep(0.005)
        total_actions = agent.survival_mode_count + agent.explore_mode_count
        survival_ratio = (agent.survival_mode_count / total_actions) if total_actions > 0 else 0
        survival_ratios.append(survival_ratio)
        scores.append(state['score'])
        steps_list.append(steps)
        if env.game.pellets.isEmpty():
            completion_rate += 1
    return {
        'scores': scores,
        'steps': steps_list,
        'survival_ratios': survival_ratios,
        'completion_rate': (completion_rate / episodes) * 100
    }

def comparative_analysis(episodes=10):
    env = SurvivalBasedPacmanWrapper()
    initial_state = env.reset()
    state_size = len(env.extract_features(initial_state))
    action_size = ACTION_DIM

    agents = {}
    for agent_name in ['survival', 'ppo', 'q_learning']:
        agent = SurvivalBasedAgent(state_size, action_size)
        try:
            agent.load_agents()
        except:
            print(f"Error loading models for {agent_name}")
        if agent_name == 'ppo':
            agent.threat_threshold = -1
        if agent_name == 'q_learning':
            agent.threat_threshold = float('inf')
        agents[agent_name] = agent

    results = {}
    for name, agent in agents.items():
        print(f"Evaluating {name} agent...")
        results[name] = evaluate(agent, episodes)
    return results

def ablation_study(episodes=10):
    env = SurvivalBasedPacmanWrapper()
    initial_state = env.reset()
    state_size = len(env.extract_features(initial_state))
    action_size = ACTION_DIM

    variants = {}

    base = SurvivalBasedAgent(state_size, action_size)
    base.load_agents()
    variants['base'] = base

    low_thresh = SurvivalBasedAgent(state_size, action_size)
    low_thresh.load_agents()
    low_thresh.threat_threshold = 0.025
    variants['low_thresh'] = low_thresh

    high_thresh = SurvivalBasedAgent(state_size, action_size)
    high_thresh.load_agents()
    high_thresh.threat_threshold = 0.1
    variants['high_thresh'] = high_thresh

    uniform = SurvivalBasedAgent(state_size, action_size)
    uniform.load_agents()
    uniform.ghost_weights = [1, 1, 1, 1]
    variants['uniform'] = uniform

    results = {}
    for name, agent in variants.items():
        print(f"Evaluating {name} variant...")
        results[name] = evaluate(agent, episodes)
    return results

# --- Main Analysis ---

def main():
    os.makedirs("evaluation/plots", exist_ok=True)

    # Comparative Analysis
    comparative = comparative_analysis(episodes=10)
    print("\nFinished Comparative Analysis.")

    # Ablation Study
    ablation = ablation_study(episodes=10)
    print("\nFinished Ablation Study.")

    # Prepare DataFrames
    comp_df = pd.DataFrame()
    for agent, data in comparative.items():
        for i in range(len(data['scores'])):
            comp_df = pd.concat([comp_df, pd.DataFrame({
                'Agent': [agent],
                'Score': [data['scores'][i]],
                'Steps': [data['steps'][i]],
                'Survival Ratio': [data['survival_ratios'][i]]
            })])

    ablation_df = pd.DataFrame()
    for variant, data in ablation.items():
        for i in range(len(data['scores'])):
            ablation_df = pd.concat([ablation_df, pd.DataFrame({
                'Variant': [variant],
                'Score': [data['scores'][i]],
                'Steps': [data['steps'][i]],
                'Survival Ratio': [data['survival_ratios'][i]]
            })])

    # Plot Comparative
    plt.figure(figsize=(8,6))
    sns.barplot(x='Agent', y='Score', data=comp_df, ci='sd', capsize=.2)
    plt.title('Comparative Analysis - Scores')
    plt.savefig('evaluation/plots/comparative_scores.png')
    plt.close()

    plt.figure(figsize=(8,6))
    sns.barplot(x='Agent', y='Steps', data=comp_df, ci='sd', capsize=.2)
    plt.title('Comparative Analysis - Steps')
    plt.savefig('evaluation/plots/comparative_steps.png')
    plt.close()

    # Plot Ablation
    plt.figure(figsize=(8,6))
    sns.barplot(x='Variant', y='Score', data=ablation_df, ci='sd', capsize=.2)
    plt.title('Ablation Study - Scores')
    plt.savefig('evaluation/plots/ablation_scores.png')
    plt.close()

    plt.figure(figsize=(8,6))
    sns.barplot(x='Variant', y='Steps', data=ablation_df, ci='sd', capsize=.2)
    plt.title('Ablation Study - Steps')
    plt.savefig('evaluation/plots/ablation_steps.png')
    plt.close()

    # Statistical testing
    print("\nStatistical Tests (Survival-Based vs Others):")
    surv_scores = comparative['survival']['scores']
    ppo_scores = comparative['ppo']['scores']
    q_scores = comparative['q_learning']['scores']

    ttest_surv_ppo = stats.ttest_rel(surv_scores, ppo_scores)
    ttest_surv_q = stats.ttest_rel(surv_scores, q_scores)

    print(f"Survival-Based vs PPO-Only: p-value = {ttest_surv_ppo.pvalue:.4f}")
    print(f"Survival-Based vs Q-Learning-Only: p-value = {ttest_surv_q.pvalue:.8f}")

if __name__ == "__main__":
    main()
