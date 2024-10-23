import logging
import pickle
import os
import csv
import pandas as pd
from termcolor import colored

import torch
import numpy as np
from tensordict import TensorDict
from tqdm import tqdm
import wandb
import random
import string
from omegaconf import OmegaConf


def benchmark_setting(config):
    """Set the benchmarking setting for the evaluation."""
    OmegaConf.update(config, 'seed', 42)
    OmegaConf.update(config, 'eval_n_eps', 30)
    OmegaConf.update(config, 'eval_n_parallel_envs', 1)
    OmegaConf.update(config, 'sim.n_agents', 10)
    OmegaConf.update(config, 'sim.start_pose_mode', 'sequential')
    OmegaConf.update(config, 'sim.start_positions.dynamic_reset', False)
    OmegaConf.update(config, 'sim.start_positions.n_split', None)
    OmegaConf.update(config, 'planner.opponents.params.vgain', [0.75])
    return config


class RolloutRecording:
    """List like object that stores a history of recorded episodes.

    Attributes:
        map_name (str): Name of the environment the episodes are recorded in
        history (list): Data storage of the recordings
    """

    def __init__(self, map_name):
        self.map_name = map_name
        self.history = [{'obs': [], 'action': [], 'reward': []}]

    def append(self, new_episode: bool, obs, action, reward):
        """Operator to append the recordings to the data storage."""
        self.history[-1]['obs'].append(obs)
        self.history[-1]['action'].append(action)
        self.history[-1]['reward'].append(reward)

        # The next set added will be the start of a new eps
        if new_episode:
            self.history.append({'obs': [], 'action': [], 'reward': []})


def save_agent_to_wandb(config, agent, optimizer, obs_rms, metadata={}, aliases=["latest"]):
    """Save the state_dict of the agents.py to disk."""
    model_dict = {
        'model_state_dict': agent.state_dict(),
        'obs_rms': obs_rms,
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'config': config
    }
    torch.save(model_dict, f'{wandb.run.dir}/agent.pt')

    model_artifact = wandb.Artifact(
        f"race_car-{wandb.run.id}", type="model", description="Race car model trained by RPL",
        metadata=metadata
    )
    if config.wandb:
        model_artifact.add_file(f'{wandb.run.dir}/agent.pt')
        artifact = wandb.log_artifact(model_artifact, aliases=aliases)  # if is_best else None) "best"
        os.remove(f'{wandb.run.dir}/agent.pt')
    else:
        print(f"Model saved to {wandb.run.dir}/agent.pt")


@torch.inference_mode()
def evaluate_agent(config, agent, envs, obs_rms):
    """ Evaluate the agents.py in the environment."""
    assert config.eval_n_eps % config.eval_n_parallel_envs == 0
    eval_n_eps = config.eval_n_eps // config.eval_n_parallel_envs

    if obs_rms is not None:
        logging.warning('Using observation normalization for evaluation.')
        envs.set_obs_rms(obs_rms)

    agent.train(False)
    envs.set_block_update_obs(True)

    logging.info(f'Evaluating agent for {eval_n_eps} episodes per map...')
    vec_eval(config, envs, agent, eval_n_eps, sample_mode=config.sample_mode, global_step=None)
    logging.info(f'Evaluation finished.')

    envs.set_block_update_obs(False)
    agent.train(True)


def vec_eval(config, envs, agent, eval_n_eps, sample_mode, global_step=None):
    """ Evaluate the agents.py in the environment."""
    avg_finish_time = []
    n_crash = 0
    n_crash_overtaking = 0
    n_crash_env = 0
    n_overtook = 0
    n_distance_driven = 0
    records_all = []

    map_names = envs.map_names
    map_types = envs.map_types
    histories = vec_rollout(envs, agent, map_names, eval_n_eps, sample_mode)

    # make a pandas dataframe to store the results
    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    results_train.index.name = 'Map'
    results_test.index.name = 'Map'

    # Create records for all episodes and maps in the history
    for map_name_, map_type_, history in zip(map_names, map_types, histories):
        history_ = history.history
        for episode, history_eps in enumerate(history_):
            # Fetch the observations and actions
            obs_all = dict()
            for key in history_eps['obs'][0].keys():
                # Flatten list into dicts
                obs_all[key] = np.array([o[key] for o in history_eps['obs']])
            obs_all['reward'] = np.array(history_eps['reward'])
            obs_all['action_applied'] = np.array(history_eps['action'])
            obs_all['action_planner'] = obs_all['action_planner']
            obs_all['action_residual'] = obs_all['action_applied'] - obs_all['action_planner']

            # Process the observations
            records = get_records_episode(obs_all)

            if not records["metrics"]['collisions']:
                avg_finish_time.append(records["metrics"]["best_finish_time"])
            n_crash += records["metrics"]['collisions']
            n_crash_overtaking += records["metrics"]['collisions_overtaking_agents'] + records["metrics"][
                'collisions_overtaking_env']
            n_crash_env += records["metrics"]['collisions_env']
            n_overtook += records["metrics"]['overtaking_success_sum']
            n_distance_driven += records["metrics"]['distance_driven']
            records_all.append(records)

            df = pd.DataFrame(records['metrics'], index=[map_name_])
            if map_type_ == 'train':
                results_train = pd.concat([results_train, df])
            elif map_type_ == 'test':
                results_test = pd.concat([results_test, df])

            # Local dump of the records
            hash_code = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
            log_dir = f'{config.log_dir}/eval/{global_step}/{map_name_}/{hash_code}/'
            # check if the directory exists
            dir_exists = os.path.exists(log_dir)
            while dir_exists:
                hash_code = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
                log_dir = f'{config.log_dir}/eval/{global_step}/{map_name_}/{hash_code}/'
                if not os.path.exists(log_dir):
                    dir_exists = False
            os.makedirs(log_dir, exist_ok=False)
            with open(f'{log_dir}records.pkl', 'wb') as f:
                pickle.dump(records, f, protocol=3)
            with open(f"{log_dir}history.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=records['history'].keys())
                writer.writeheader()
                for row in range(records['history']['action_residual_steer'].shape[0]):
                    # Note: Obs and action have different length, i.e., the real last obs (= when done) is missing
                    writer.writerow({k: v[row] for k, v in records['history'].items() if k != 'finish_times'})
            with open(f"{log_dir}metrics.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=records['metrics'].keys())
                writer.writeheader()
                writer.writerow(records['metrics'])

    # Evaluate the results over all maps and episodes
    results_idx_selected = [
        'best_finish_time',
        # 'total_return',
        # 'overtaking_success_sum',
        # 'collisions',
        'collision_rate_overtaking',
        # 'collision_rate_overtaking_agent',
        # 'collision_rate_overtaking_env',
        'collision_per_driven_km',
        'track_type'
    ]

    # All results
    results_all = pd.concat([results_train, results_test])
    results_all_mean = evaluate_results_pd(results_all)

    track_type_dict = dict()
    track_type_dict.update({k: 'train' for k in results_train.index})
    track_type_dict.update({k: 'test' for k in results_test.index})

    # add the info if racetrack is train or test depending on the index. test if the track is also in the test set
    if len(results_train) > 0:
        results_train_mean = evaluate_results_pd(results_train)
        results_train_mean['track_type'] = 'train'
    if len(results_test) > 0:
        results_test_mean = evaluate_results_pd(results_test)
        results_test_mean['track_type'] = 'test'

    results_all_mean['track_type'] = results_all_mean.index.map(track_type_dict)

    if len(results_train) > 0:
        results_all_mean.loc['All_train'] = results_train_mean.loc['All']
    if len(results_test) > 0:
        results_all_mean.loc['All_test'] = results_test_mean.loc['All']

    # safe them to pickle
    wandb.log({'results_all_mean': wandb.Table(dataframe=results_all_mean.reset_index())})
    results_all_mean.to_pickle(f'{config.log_dir}/eval/{global_step}/results_all.pkl')
    print(colored(f'\nMean results per all maps:', 'blue'))
    print(f'{results_all_mean[results_idx_selected].round(4)}')

    logging.info(f'\nSUMMARY:')
    logging.info(f'- Overtaking successes: {n_overtook}')
    logging.info(f'- Total crashes: {n_crash} | crash rate: {round(n_crash / (n_overtook + n_crash), 4)}')
    logging.info(f'- Env crashes: {n_crash_env}  | crash/distance: {round(n_crash_env / n_distance_driven, 4)}')
    logging.info(f'- Overtaking crashes (agent + env): {n_crash_overtaking} '
                 f'| crash rate: {round(n_crash_overtaking / (n_overtook + n_crash_overtaking), 4)}')


def vec_rollout(envs, agent, map_name=None, eval_n_eps=1, sample_mode=True):
    """Parallel rollout of the agents.py in vectorized envs.

    Note: Vector Env will reset itself automatically.

    Args:
        envs (VecEnv): Vectorized environment.
        agent (Agent): Agent to rollout.
        action_scaler (list): Scaling factor for the actions.
        map_name (list): List of map names.
        n_eval (int): Number of episodes to rollout.

    Returns:
        list: List of RolloutRecording objects, one for each map.
    """
    bar = tqdm(total=eval_n_eps)
    with torch.no_grad():
        recordings = [RolloutRecording(n) for n in map_name]

        obs_wrapped, infos = envs.reset()
        obs_wrapped = TensorDict(obs_wrapped, [obs_wrapped['acc_x'].shape[0]], device=agent.device)
        new_observations = infos['obs_']  # Access the non-normalized observations through infos
        next_done = torch.zeros(envs.num_envs, device=agent.device, dtype=torch.bool)

        while True:
            action = agent.get_action(obs_wrapped, next_done, mode=sample_mode)
            action = action.cpu().detach().numpy()
            old_observations = new_observations
            obs_wrapped, rewards, terminateds, truncateds, infos = envs.step(action)
            next_terminated = torch.tensor(terminateds, device=agent.device)
            next_truncated = torch.tensor(truncateds, device=agent.device)
            next_done = torch.logical_or(next_terminated, next_truncated)

            obs_wrapped = TensorDict(obs_wrapped, [obs_wrapped['acc_x'].shape[0]], device=agent.device)
            new_observations = infos['obs_']

            for i in range(len(map_name)):
                recordings[i].append(
                    False,
                    old_observations[i],
                    action[i, :],
                    rewards[i]
                )

            if 'final_info' in infos:
                for i in range(len(map_name)):
                    if infos['_final_info'][i]:
                        recordings[i].append(
                            True,
                            infos['final_info'][i]['obs_'],
                            np.array([np.nan, np.nan]),
                            np.nan
                        )

            if all([len(h.history) > bar.n + 1 for h in recordings]):
                bar.update(1)
                if all([len(h.history) > eval_n_eps for h in recordings]):
                    for h in recordings:
                        h.history = h.history[:eval_n_eps]
                    break
    bar.close()
    return recordings


def get_records_episode(obs_all):
    """Process the recorded observations to obtain statistics of them."""
    records = dict()

    # History of states
    history = dict()
    history['poses_x'] = obs_all['poses_x'].squeeze()
    history['poses_y'] = obs_all['poses_y'].squeeze()
    history['poses_theta'] = obs_all['poses_theta'].squeeze()
    history['close_opponent_x'] = obs_all['close_opponent'].squeeze()[:, 0]
    history['close_opponent_y'] = obs_all['close_opponent'].squeeze()[:, 1]
    history['close_opponent_theta'] = obs_all['close_opponent'].squeeze()[:, 2]
    # history['lookahead_points_relative_x'] = obs_all['lookahead_points_relative'][:, 0]
    # history['lookahead_points_relative_y'] = obs_all['lookahead_points_relative'][:, 1]
    history['linear_vels_x'] = obs_all['linear_vels_x'].squeeze()
    history['linear_vels_y'] = obs_all['linear_vels_y'].squeeze()
    history['ang_vels'] = obs_all['ang_vels_z'].squeeze()
    history['slip_angle'] = obs_all['slip_angle'].squeeze()
    history['acc_x'] = obs_all['acc_x'].squeeze()
    history['acc_y'] = obs_all['acc_y'].squeeze()
    history['rewards'] = obs_all['reward'].squeeze()
    history['collisions'] = obs_all['collisions'].squeeze()
    history['collisions_env'] = obs_all['collisions_env'].squeeze()
    history['collisions_overtaking_agents'] = obs_all['collisions_overtaking_agents'].squeeze()
    history['collisions_overtaking_env'] = obs_all['collisions_overtaking_env'].squeeze()
    history['lap_times'] = obs_all['lap_times'].squeeze()
    history['lap_counts'] = obs_all['lap_counts'].squeeze()
    history['overtaking_success'] = obs_all['overtaking_success'].squeeze()
    history['finish_times'] = [history['lap_times'][history['lap_counts'] == i + 1].min() for i in
                               range(0, history['lap_counts'].max()) if
                               not any(history['collisions'][history['lap_counts'] == i + 1])]

    # # History of actions
    history['action_residual_steer'] = obs_all['action_residual'].squeeze()[:, 0]
    history['action_residual_vel'] = obs_all['action_residual'].squeeze()[:, 1]
    history['action_planner_steer'] = obs_all['action_planner'][:obs_all['action_residual'].shape[0], 0]
    history['action_planner_vel'] = obs_all['action_planner'][:obs_all['action_residual'].shape[0], 1]
    # Applied action is really the action thas been used in the env -> actions may be clipped!
    history['action_applied_steer'] = obs_all['action_applied'][:, 0]
    history['action_applied_vel'] = obs_all['action_applied'][:, 1]

    # Add to list
    records['history'] = history

    # History of performance
    metrics = dict()
    # Analysis of actions and stats
    for metrics_name in ['action_residual_vel', 'action_applied_vel', 'action_residual_steer', 'action_applied_steer',
                         'linear_vels_x']:
        metrics[f'{metrics_name}_mean'] = np.nanmean(history[f'{metrics_name}'], 0)
        metrics[f'{metrics_name}_median'] = np.nanmedian(history[f'{metrics_name}'], 0)
        metrics[f'{metrics_name}_std'] = np.nanstd(history[f'{metrics_name}'], 0)
        metrics[f'{metrics_name}_max'] = np.nanmax(history[f'{metrics_name}'], 0)
        metrics[f'{metrics_name}_min'] = np.nanmin(history[f'{metrics_name}'], 0)

    for metrics_name in ['action_residual_steer', 'action_applied_steer', 'linear_vels_y', 'slip_angle']:
        metrics[f'{metrics_name}_abs_mean'] = np.nanmean(np.abs(history[f'{metrics_name}']), 0)
        metrics[f'{metrics_name}_abs_median'] = np.nanmedian(np.abs(history[f'{metrics_name}']))
        metrics[f'{metrics_name}_abs_std'] = np.nanstd(np.abs(history[f'{metrics_name}']), 0)
        metrics[f'{metrics_name}_abs_max'] = np.nanmax(np.abs(history[f'{metrics_name}']), 0)
        metrics[f'{metrics_name}_abs_min'] = np.nanmin(np.abs(history[f'{metrics_name}']), 0)

    # Performance metrics
    metrics['overtaking_success_sum'] = history['overtaking_success'].sum()
    metrics['rewards_mean'] = np.nanmean(history['rewards'])
    metrics['rewards_std'] = np.nanstd(history['rewards'])
    metrics['total_return'] = np.nansum(history['rewards'])
    metrics['collisions'] = history['collisions'].sum(0)
    metrics['collisions_env'] = history['collisions_env'].sum(0)
    metrics['collisions_overtaking_agents'] = history['collisions_overtaking_agents'].sum(0)
    metrics['collisions_overtaking_env'] = history['collisions_overtaking_env'].sum(0)
    metrics['steps'] = history['rewards'].shape[0]  # 1 Step less than the num of observed states due to 'done'
    metrics['full_laps'] = history['lap_counts'].max()
    metrics['full_laps_sum'] = history['lap_counts'].sum()

    # calculate distance driven
    distance_driven = 0.0
    for i in range(1, len(history['poses_x'])):
        distance_driven += np.sqrt((history['poses_x'][i] - history['poses_x'][i - 1]) ** 2 +
                                   (history['poses_y'][i] - history['poses_y'][i - 1]) ** 2)
    metrics['distance_driven'] = distance_driven / 1000.0  # in km

    # Lap times are recorded for a maximum of 2 laps
    if len(history['finish_times']) < 2:
        metrics['best_finish_time'] = 0.0
    else:
        # Finish time is the lap times of the second lap (running start)
        metrics['best_finish_time'] = history['finish_times'][1] - history['finish_times'][0]

    # Add to list
    records['metrics'] = metrics

    return records


def evaluate_results_pd(results):
    """Evaluate the results of the evaluation."""
    results_mean = results.groupby(level=0).mean()

    results_mean['best_finish_time'] = results.groupby(level=0)['best_finish_time'].apply(lambda x: x[x > 0].median())

    # Overtaking success rate
    results_mean['collision_rate_overtaking_agent'] = results_mean['collisions_overtaking_agents'] / (
            results_mean['overtaking_success_sum'] + results_mean['collisions_overtaking_agents'])
    collision_rate_overtaking_agent_all = results_mean['collisions_overtaking_agents'].sum() / (
            results_mean['overtaking_success_sum'] + results_mean['collisions_overtaking_agents']).sum()

    results_mean['collision_rate_overtaking_env'] = results_mean['collisions_overtaking_env'] / (
            results_mean['overtaking_success_sum'] + results_mean['collisions_overtaking_env'])
    collision_rate_overtaking_env_all = results_mean['collisions_overtaking_env'].sum() / (
            results_mean['overtaking_success_sum'] + results_mean['collisions_overtaking_env']).sum()

    results_mean['collision_rate_overtaking'] = (results_mean['collisions_overtaking_env'] + results_mean[
        'collisions_overtaking_agents']) / (results_mean['overtaking_success_sum'] + results_mean[
        'collisions_overtaking_env'] + results_mean['collisions_overtaking_agents'])
    collision_rate_overtaking_all = (results_mean['collisions_overtaking_env'] + results_mean[
        'collisions_overtaking_agents']).sum() / (results_mean['overtaking_success_sum'] + results_mean[
        'collisions_overtaking_env'] + results_mean['collisions_overtaking_agents']).sum()

    # Env collisions per driven km
    results_mean['collision_per_driven_km'] = results_mean['collisions_env'] / results_mean['distance_driven']
    collision_per_driven_km_all = results_mean['collisions_env'].sum() / results_mean['distance_driven'].sum()

    results_mean.loc['All'] = results_mean.mean()
    results_mean.loc['All', 'collision_rate_overtaking_agent'] = collision_rate_overtaking_agent_all
    results_mean.loc['All', 'collision_rate_overtaking_env'] = collision_rate_overtaking_env_all
    results_mean.loc['All', 'collision_rate_overtaking'] = collision_rate_overtaking_all
    results_mean.loc['All', 'collision_per_driven_km'] = collision_per_driven_km_all

    return results_mean
