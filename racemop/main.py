import os
import logging
import random
import hydra
from hydra.core.hydra_config import HydraConfig, DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import numpy as np
import torch
from tensordict import TensorDict
from copy import deepcopy

from racemop import maker
from racemop.agents.loss import PPOLoss
from racemop.agents.nn_utils import save_model_summary
from racemop.env_wrapper import RewardLogger
from racemop.evaluation import save_agent_to_wandb, evaluate_agent, benchmark_setting


@hydra.main(version_base=None, config_path='../configs/', config_name='config')
def main(config: DictConfig) -> None:
    """Main function to run the training.

    Args:
        config (DictConfig): Configuration object.
    """

    # Seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Get some speed-up by using TensorCores on Nvidia Ampere GPUs, can cause issues with reproducibility
    torch.backends.cudnn.deterministic = config.torch_deterministic
    torch.backends.cudnn.allow_tf32 = config.cuda_allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = config.cuda_allow_tf32
    torch.backends.cudnn.benchmark = True
    device = torch.device(config.device)

    # Update config
    OmegaConf.update(config, 'log_dir', HydraConfig.get().run.dir, force_add=True)
    OmegaConf.update(config, 'batch_size', int(config.rl.num_envs * config.rl.num_steps), force_add=True)
    OmegaConf.update(config, 'minibatch_size', int(config.batch_size // config.rl.num_minibatches), force_add=True)
    OmegaConf.update(config, 'num_updates', int(config.rl.total_timesteps // config.batch_size), force_add=True)

    wandb.init(
        project='raceMOP',
        entity='tumwcps',
        sync_tensorboard=True,
        config=OmegaConf.to_container(config, resolve=True),
        dir=config.log_dir,
        mode='online' if config.wandb else 'disabled',
        save_code=True,
        tags=["frame_stacking"],
        notes=config.notes,
        settings=wandb.Settings(code_dir=".")
    )
    os.environ["WANDB_SILENT"] = "true"

    # Allows to get some reference values of the baseline without residual actions applied
    if config.mode == 'baseline':
        logging.info(f"Running in baseline mode.")
        from racemop.agents.agents import BaseControllerAgent
        # Baseline planner passes planned action through to env
        if config.benchmark:
            config = benchmark_setting(config)
            wandb.config.update(OmegaConf.to_container(config, resolve=True), allow_val_change=True)
        envs, action_space, obs_space = maker.make_eval_envs(config)
        agent = BaseControllerAgent(config, action_space, obs_space)
        evaluate_agent(config, agent, envs, obs_rms=None)
        exit(0)

    elif config.mode == 'inference':
        logging.info(f"Running in {config.mode} mode.")
        # Fetch the model from wandb
        if config.wandb:
            artifact = wandb.run.use_artifact(f'{config.wandb_model.name}:{config.wandb_model.alias}')
            artifact_dir = artifact.download(root=wandb.run.dir)
            logging.info(f"Loaded agent {artifact.source_name} (global step {artifact.metadata['global_step']}).")
            checkpoint = torch.load(f'{artifact_dir}/agent.pt', map_location=device)
            os.remove(f'{wandb.run.dir}/agent.pt')
        else:
            checkpoint = torch.load('/tmp/agent.pt', map_location=device)

        # Update config with the saved one from wandb
        OmegaConf.update(config, 'rl', checkpoint['config'].rl)
        OmegaConf.update(config, 'env', checkpoint['config'].env)
        OmegaConf.update(config, 'planner.ego', checkpoint['config'].planner.ego)

        if config.benchmark:
            config = benchmark_setting(config)
        wandb.config.update(OmegaConf.to_container(config, resolve=True), allow_val_change=True)

        # Make envs
        envs, action_space, obs_space = maker.make_eval_envs(config)

        # Create the agent and initialize it with the weights from wandb
        agent, next_done = maker.make_agent(config, action_space, obs_space, device)
        if config.compile_torch:
            agent = torch.compile(agent, mode='max-autotune')

        agent.load_state_dict(checkpoint['model_state_dict'])
        agent = agent.to(device)

        # Evaluate the agent
        evaluate_agent(config, agent, envs, obs_rms=checkpoint['obs_rms'])
        exit(0)

    elif config.mode == 'training':
        logging.info(
            f'\nRunning in training mode with the following configuration:'
            f'\n- Learning device: {device}'
            f'\n- Num_updates: {config.num_updates}'
            f'\n- Update epochs: {config.rl.update_epochs}'
            f'\n- Full batch size: {config.batch_size}'
            f'\n- Mini batch size: {config.minibatch_size}\n'
        )

        # Make envs
        envs, action_space, obs_space = maker.make_training_envs(config)

    # TRAINING MODE
    # Agent
    agent, next_done = maker.make_agent(config, action_space, obs_space, device)

    # Loss
    ppo_loss = PPOLoss(config).to(device)
    gamma = torch.tensor(config.rl.gamma, device=device, requires_grad=False)
    gae_lambda = torch.tensor(config.rl.gae_lambda, device=device, requires_grad=False)

    # Init lazy layers
    with torch.no_grad():
        next_obs, _ = envs.reset(seed=config.seed)
        next_obs = TensorDict(next_obs, [next_obs['acc_x'].shape[0]], device=device)
        agent.get_action_and_value(next_obs, None, next_done)
        save_model_summary(log_dir=wandb.run.dir, models=[agent], input_data=[next_obs])
        wandb.save("model_summary.txt")

    if config.compile_torch:
        # Must be run after the summary
        agent = torch.compile(agent, mode='max-autotune')
        ppo_loss = torch.compile(ppo_loss, mode='max-autotune')

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=config.rl.learning_rate,
        eps=1e-5,
        weight_decay=config.rl.weight_decay
    )

    if config.rl.anneal_lr:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_updates + 1,
            eta_min=config.rl.anneal_lr_factor * optimizer.param_groups[0]["lr"]
        )

    # 4) Storage setup
    batch_dim = (config.rl.num_steps, config.rl.num_envs)
    obs = TensorDict({}, batch_dim, device=device)
    actions = torch.zeros(batch_dim + envs.get_wrapper_attr('single_action_space').shape).to(device)
    logprobs = torch.zeros(batch_dim, device=device)
    rewards = torch.zeros(batch_dim, device=device)
    values = torch.zeros(batch_dim, device=device)
    terminateds = torch.zeros(batch_dim, device=device, dtype=torch.bool)  # terminates
    truncateds = torch.zeros(batch_dim, device=device, dtype=torch.bool)
    dones = torch.zeros(batch_dim, device=device, dtype=torch.bool)

    # Pretrain the critic if desired
    pretrain_reset_needed = False
    if config.rl.critic_safe_exploration and config.rl.critic_pretrain > 0.0:
        agent.actor_logstd.data = torch.tensor([-3.0, -3.0], device=agent.device, requires_grad=True)
        pretrain_reset_needed = True

    # Logger for the rewards terms
    reward_logger = RewardLogger(config)
    best_mean_reward = -np.inf
    best_collision_reward = -np.inf

    # Training setup and progress bars
    global_step = 0
    i_grad_step = 0
    i_pretrain = 0
    bar_update = tqdm(range(1, config.num_updates + 1), desc='Update steps', colour='yellow', position=0)
    bar_data = tqdm(range(0, config.rl.num_steps), desc='Data collection', colour='blue', position=1)
    bar_epoch = tqdm(range(config.rl.update_epochs), desc='Training epochs', colour='green', position=2)

    # GO!
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = TensorDict(next_obs, [next_obs['acc_x'].shape[0]], device=device)
    next_terminated = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
    next_truncated = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
    next_done = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)

    for update in range(1, config.num_updates + 1):
        if config.env.wrapper.block_updates and update / config.num_updates > config.env.wrapper.block_after_n_updates:
            envs.set_block_update_rew(True)
            envs.set_block_update_obs(True)
            assert envs.block_update_rew and envs.block_update_obs, "Updating normalization stats couldn't be blocked."

        bar_data.reset()
        for step in range(0, config.rl.num_steps):
            global_step += 1 * config.rl.num_envs
            obs[step] = next_obs
            terminateds[step] = next_terminated
            truncateds[step] = next_truncated
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, action_combined, logprob, _, value = agent.get_action_and_value(
                    next_obs, None, next_done
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # check actions for nan
            if torch.isnan(action_combined).any() or torch.isnan(action).any():
                raise ValueError("Action is NaN!")

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action_combined.cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_terminated = torch.tensor(terminated, device=device)
            next_truncated = torch.tensor(truncated, device=device)
            next_done = torch.logical_or(next_terminated, next_truncated)
            next_obs = TensorDict(next_obs, [next_obs['acc_x'].shape[0]], device=device)
            # Raw rewards in infos for logging
            reward_logger(step, infos)
            bar_data.update(1)

        if config.rl.pretrain and i_pretrain < 1:
            i_pretrain += 1
            agent.modality_encoder.pretrain(obs)

        # bootstrap value if not done
        with torch.no_grad():
            agent.train(False)
            next_value = agent.get_value(next_obs, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.rl.num_steps)):
                if t == config.rl.num_steps - 1:
                    nextnonterminal = ~next_terminated
                    nextnondone = ~next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = ~terminateds[t + 1]
                    nextnondone = ~dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal.float() - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnondone.float() * lastgaelam
                if advantages.abs().max() > 1000:
                    raise ValueError("Advantage is exploding!")
            returns = advantages + values
            agent.train(True)

        b_obs = obs.view(-1)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.get_wrapper_attr('single_action_space').shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Get some stats for the batch
        v_pred, v_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_v = np.var(v_true)
        explained_var = np.nan if var_v == 0 else 1 - np.var(np.abs(v_true - v_pred)) / var_v
        rmse_v = np.sqrt(np.mean((v_true - v_pred) ** 2))
        if np.abs(rmse_v) > 1000:
            raise ValueError("RMSE is exploding!")

        mae_v = np.mean(np.abs(v_true - v_pred))
        # largest prediction error
        max_error_v = np.max(np.abs(v_true - v_pred))
        max_v_pred = np.max(v_pred)
        min_v_pred = np.min(v_pred)
        mean_v_pred = np.mean(v_pred)
        max_v_true = np.max(v_true)
        min_v_true = np.min(v_true)
        mean_v_true = np.mean(v_true)

        try:
            rew_stats = envs.get_rew_rms()
            return_rms_mean = rew_stats.mean
            return_rms_std = rew_stats.var
            return_rms_count = rew_stats.count
        except:
            return_rms_mean = 0.0
            return_rms_std = 0.0
            return_rms_count = 0.0

        mean_reward = reward_logger.running_mean_rewards.calculate_mean()
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            obs_rms = deepcopy(envs.get_wrapper_attr('get_obs_rms')())
            metadata = {'best_mean_reward': best_mean_reward, 'global_step': global_step}
            save_agent_to_wandb(config, agent, optimizer, obs_rms, metadata, aliases=['best_reward'])
            wandb.run.summary["best_mean_reward"] = best_mean_reward

        mean_reward_collisions = reward_logger.running_mean_rewards_collisions.calculate_mean()
        bar_update.set_postfix({'r_c': f'{mean_reward_collisions:.3f}'})

        if mean_reward_collisions > best_collision_reward:
            best_collision_reward = mean_reward_collisions
            obs_rms = deepcopy(envs.get_wrapper_attr('get_obs_rms')())
            metadata = {'best_collision_reward': best_collision_reward, 'global_step': global_step}
            save_agent_to_wandb(config, agent, optimizer, obs_rms, metadata, aliases=['best_collision'])
            wandb.run.summary["best_collision_reward"] = best_collision_reward

        # if nan explained_var
        if np.isnan(explained_var):
            logging.info('Explained var is NaN! Enforcing reset.')
            next_obs, _ = envs.reset(seed=config.seed)
            next_obs = TensorDict(next_obs, [next_obs['acc_x'].shape[0]], device=device)
            next_terminated = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
            next_truncated = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
            next_done = torch.zeros(config.rl.num_envs, device=device, dtype=torch.bool)
            continue

        wandb.log({
            "batch/explained_variance": explained_var,
            "batch/rmse_v": rmse_v,
            "batch/mae_v": mae_v,
            "batch/max_error_v": max_error_v,
            "batch/max_v_pred": max_v_pred,
            "batch/min_v_pred": min_v_pred,
            "batch/mean_v_pred": mean_v_pred,
            "batch/max_v_true": max_v_true,
            "batch/min_v_true": min_v_true,
            "batch/mean_v_true": mean_v_true,
            "batch/action[0]_mean": b_actions[:, 0].mean(),
            "batch/action[0]_std": b_actions[:, 0].std(),
            "batch/action[1]_mean": b_actions[:, 1].mean(),
            "batch/action[1]_std": b_actions[:, 1].std(),
            # "batch/collision_mean": b_obs['collision'].mean(),
            "others/actor_std[0]": torch.exp(agent.get_parameter('actor_logstd').detach().cpu()).numpy()[0],
            "others/actor_std[1]": torch.exp(agent.get_parameter('actor_logstd').detach().cpu()).numpy()[1],
            "others/return_rms_mean": return_rms_mean,
            "others/return_rms_std": return_rms_std,
            "others/return_rms_count": return_rms_count,
            "others/best_mean_reward": best_mean_reward,
            "others/global_step": global_step,
        })

        clipfracs = []
        b_inds = np.arange(config.batch_size)
        batch_size = config.batch_size
        minibatch_size = config.minibatch_size
        sub_minibatch_size = config.rl.num_steps  # Skip TBPTT as one interation

        bar_epoch.reset()

        for epoch in range(config.rl.update_epochs):
            np.random.shuffle(b_inds)
            approx_kl = 0.0
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                for sub_start in range(0, config.rl.num_steps, sub_minibatch_size):
                    _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        x=b_obs[mb_inds],
                        action=b_actions[mb_inds],
                        done=b_dones[mb_inds]
                    )

                    loss, pg_loss, v_loss, entropy_loss, approx_kl, old_approx_kl, clipfrac, grads = \
                        ppo_loss.update(
                            config, agent, optimizer,
                            b_returns[mb_inds], b_values[mb_inds], b_advantages[mb_inds], b_logprobs[mb_inds],
                            newlogprob, entropy, newvalue
                        )
                    clipfracs += [clipfrac.float().mean().item()]

                    wandb.log({
                        "losses/clipfrac": np.mean(clipfracs),
                        "losses/policy_loss": pg_loss.item(),
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/value_loss": v_loss.item(),
                        "losses/newvalue_mean": newvalue.cpu().detach().numpy().mean(),
                        "losses/epochs": epoch,
                        "losses/grads": grads,
                        "losses/learning_rate": optimizer.param_groups[0]["lr"],
                        "losses/grad_step": i_grad_step
                    })

                    i_grad_step += 1

                if config.rl.target_kl is not None and approx_kl > config.rl.target_kl:
                    break

            bar_epoch.update(1)

        if config.rl.anneal_lr:
            lr_scheduler.step()
        bar_update.update(1)

    # Final evaluation for the best agent
    if config.wandb:
        artifact = wandb.run.use_artifact(f'race_car-{wandb.run.id}:best_reward')
        artifact_dir = artifact.download(root=wandb.run.dir)
        logging.info(f"Loaded agent {artifact.source_name} (global step {artifact.metadata['global_step']}).")
        checkpoint = torch.load(f'{artifact_dir}/agent.pt', map_location=device)
        os.remove(f'{wandb.run.dir}/agent.pt')
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent = agent.to(device)
        obs_rms = checkpoint['obs_rms']
    envs.close()

    # Change sim settings for evaluation
    config = benchmark_setting(config)
    envs, _, _ = maker.make_eval_envs(config)
    evaluate_agent(config, agent, envs, obs_rms=obs_rms)
    wandb.run.tags = wandb.run.tags + ('results',)

    # Clean up
    bar_update.close()
    bar_data.close()
    bar_epoch.close()
    envs.close()
