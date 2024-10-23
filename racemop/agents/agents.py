import math
import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions.normal import Normal
from torchrl.modules import TanhNormal, TruncatedNormal

from racemop.agents.nn import LidarNet, ActorMeanNet, CriticNet


class ResidualAgent(nn.Module):
    def __init__(self, config, action_space, obs_space):
        """ Base agent class
        Args:
            config: Config
            envs: Environments
        """
        super().__init__()

        self.all_frames_history = config.rl.all_frames_history

        if config.env.frame_cat.use and self.all_frames_history:
            n_frame_stack = int(math.ceil(config.env.frame_cat.n / (config.env.frame_cat.skip_slicing + 1)))
            in_features = config.rl.hidden_out_dim + config.rl.state_dim * n_frame_stack
        else:
            in_features = config.rl.hidden_out_dim + config.rl.state_dim

        # Actor
        self.actor_mean = ActorMeanNet(config, in_features=in_features)
        self.actor_logstd = nn.Parameter(torch.tensor(config.rl.init_logstd))

        # Distribution head
        if config.rl.distribution == 'TanhNormal':
            self.dist_head = TanhNormal
        elif config.rl.distribution == 'TruncatedNormal':
            self.dist_head = TruncatedNormal
        elif config.rl.distribution == 'Normal':
            self.dist_head = Normal
        else:
            raise NotImplementedError(f'Distribution {config.rl.distribution} not implemented.')

        self.register_buffer('action_space_high', torch.tensor(action_space.high, dtype=torch.float32))
        self.register_buffer('action_space_low', torch.tensor(action_space.low, dtype=torch.float32))

        if config.rl.scale_action != 'None':
            self.register_buffer('scale_action', torch.tensor(config.rl.scale_action, dtype=torch.float32))
        else:
            self.scale_action = None

        self.register_buffer('upscale_tanh', torch.tensor(config.rl.upscale_tanh))
        self.register_buffer('min_max_dist', torch.tensor([-1, 1]))
        self.tanh_loc = config.rl.tanh_loc
        self.pre_tanh_loc = config.rl.pre_tanh_loc

        # Buffer for RPO
        self.register_buffer('rpo_alpha', torch.tensor(config.rl.rpo_alpha))

        # Critic
        self.critic = CriticNet(config, in_features=in_features)

        # Perception module must be added by custom agent
        self.feature_size = None
        self.combine_before_dist = config.rl.combine_before_dist

        self.lidar_channels = obs_space['aaa_scans'].shape[0]

        self.modality_encoder = LidarNet(config, in_channel=self.lidar_channels)
        self.modality = 'aaa_scans'
        self.modality_transforms = nn.Sequential()

    @property
    def device(self):
        """ Get device of model
        Returns:
            device: Device of model
        """
        return next(self.parameters()).device

    def get_probs(self, mean, logstd):
        """ Sample from distribution.

        Args:
            mean: Mean of distribution
            logstd: Log std of distribution

        Returns:
            dist: Distribution
        """
        std = torch.exp(logstd)
        dist = self.dist_head(
            mean, std, min=self.min_max_dist[0], max=self.min_max_dist[1], upscale=self.upscale_tanh,
            tanh_loc=self.tanh_loc
        )
        return dist

    def get_features(self, x, done=None):
        """ Get features from perception module and concatenate with state
        Args:
            x: Dict of tensors
            done: Done mask
        Returns:
            features: Concatenated features
        """
        x_lidar, x_rest, a_base = self.split_data(x)
        x_transformed = self.modality_transforms(x_lidar)
        modality_encoded = self.modality_encoder.forward(x_transformed)
        features = torch.cat([x_rest, modality_encoded], dim=1)
        return self._features(features, done), a_base

    def _features(self, features, done):
        """ Get features from perception module and concatenate with state
        Args:
            features: Features
            done: Done mask
        Returns:
            features: Concatenated features
        """
        return features

    def get_value(self, x, done=None):
        """ Get value from critic
        Args:
            x: Dict of tensors
            done: Done mask
        Returns:
            value: Value
        """
        hidden, _ = self.get_features(x, done=done)
        return self.critic(hidden)

    def _combine_actions(self, action, x_base):
        """Combines the action with the base action.

        Args:
            action (torch.Tensor): Action.
            x_base (torch.Tensor): Base action.

        Returns:
            torch.Tensor: Combined action.
        """
        if self.scale_action is not None:
            return action * self.scale_action + x_base
        else:
            return action + x_base

    def get_action_and_value(self, x, action=None, done=None, action_offset=None):
        """ Get action and value from actor and critic
        Args:
            x: Dict of tensors
            action: Action to take
            done: Done mask
            action_offset: Action offset
        Returns:
            action: Action
            log_prob: Log probability of action
            entropy: Entropy of action
            value: Value
        """
        features, a_base = self.get_features(x, done)

        action_mean = self.actor_mean(features)

        if self.combine_before_dist:
            if self.scale_action is not None and type(self.dist_head) is not TanhNormal:
                if self.pre_tanh_loc:
                    action_mean = nn.Tanh()(action_mean / 10.0) * 10.0
                else:
                    action_mean = nn.Tanh()(action_mean)
            action_mean = self._combine_actions(action_mean, a_base)

        action_logstd = self.actor_logstd.expand_as(action_mean)

        if action is None:
            probs = self.get_probs(action_mean, action_logstd)
            action = probs.rsample()  # or rsample() for reparameterization trick
            if action_offset is not None:
                action_offset = torch.tensor(action_offset).to(self.device)
                action = action + action_offset.expand_as(action)

            if self.combine_before_dist:
                action_combined = action  # torch.clamp(action * self.scale_action + x_base, -1, 1)
            else:
                action_combined = self._combine_actions(action, a_base)
                action_combined = torch.clamp(action_combined, -1.0, 1.0)

        else:
            # Robust PO: For the policy update, sample again to add stochasticity
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z.to(self.device)
            probs = self.get_probs(action_mean, action_logstd)
            action_combined = None

        # The probability of the action taken/sampled
        log_prob = probs.log_prob(action)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(1)

        value = self.critic(features)

        # use the log_prob to approximate the entropy
        entropy = -log_prob.mean()

        return action, action_combined, log_prob, torch.zeros_like(log_prob), value

    def get_action(self, x, done=None, mode=True):
        """ Get action from actor
        Args:
            x: Dict of tensors
            done: Done mask
            mode: If true, return mode of distribution
        Returns:
            action: Action, not reparameterized
        """

        features, a_base = self.get_features(x, done)

        action_mean = self.actor_mean(features)

        if self.combine_before_dist:
            if self.scale_action is not None and type(self.dist_head) is not TanhNormal:
                if self.pre_tanh_loc:
                    action_mean = nn.Tanh()(action_mean / 10.0) * 10.0
                else:
                    action_mean = nn.Tanh()(action_mean)
            action_mean = self._combine_actions(action_mean, a_base)

        probs = self.get_probs(action_mean, self.actor_logstd.expand_as(action_mean))
        if mode:
            action = probs.mode
        else:
            action = probs.sample()

        if self.combine_before_dist:
            action_combined = action
        else:
            action_combined = self._combine_actions(action, a_base)
            action_combined = torch.clamp(action_combined, -1, 1)

        return action_combined

    def split_data(self, x: TensorDict):
        """Splits the input data into lidar, waypoints and rest of the states.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            x_lidar (torch.Tensor): Lidar data.
            x_wpts (torch.Tensor): Waypoints.
            x_rest (torch.Tensor): Rest of the states.
        """
        keys_not_to_include = ['aaa_scans', 'action_planner']
        if self.all_frames_history:
            if x['aaa_scans'].shape[1] == 1:
                a_base = x['action_planner']
                x_lidar = x['aaa_scans'].flatten(-1)  # remove the n_agent dim; TODO: check why scans still has it
            else:
                a_base = x['action_planner'][:, -1]
                x_lidar = x['aaa_scans'].flatten(-2)  # remove the n_agent dim; TODO: check why scans still has it
            x_rest = torch.cat(
                [x[key][:, :].flatten(1) for key in x.keys() if key not in keys_not_to_include],
                dim=1)
        else:
            if x['aaa_scans'].shape[1] == 1:
                a_base = x['action_planner']
                x_lidar = x['aaa_scans'].flatten(-1)  # remove the n_agent dim; TODO: check why scans still has it
            else:
                a_base = x['action_planner'][:, -1]
                x_lidar = x['aaa_scans'].flatten(-2)  # remove the n_agent dim; TODO: check why scans still has it
            x_rest = torch.cat(
                [x[key][:, -1].flatten(1) for key in x.keys() if key not in keys_not_to_include],
                dim=1)

        return x_lidar, x_rest, a_base

    def forward(self, x):
        return self.get_action_and_value(x, action=None, done=None, action_offset=None)


class BaseControllerAgent(ResidualAgent):
    """Agent that outputs a zero residual action."""

    def __init__(self, config, action_space, obs_space):
        super().__init__(config, action_space, obs_space)
        self.action_space = action_space
        self.action_space_low = torch.tensor(self.action_space.low, dtype=torch.float32)
        self.action_space_high = torch.tensor(self.action_space.high, dtype=torch.float32)

    def get_device(self):
        """Return the device of the agent."""
        return 'cpu'

    @property
    def device(self):
        """Return the device of the agent."""
        return self.get_device()

    def get_action(self, x, done=None, mode=True):
        """Passes the base action through without modification."""
        if len(x['action_planner'].shape) == 3:
            # Rearrange to so that the second dim is the FrameStack axis
            # x_out = x.view(-1, x.shape[-1])
            a_base = x['action_planner'][:, -1]
        else:
            a_base = x['action_planner']
        return a_base

    def train(self, mode=True):
        pass
