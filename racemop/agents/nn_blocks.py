import torch
import torch.nn as nn


class ResidualBlock1d(nn.Module):
    """Residual block for 1D convolutional neural networks."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0):
        super(ResidualBlock1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(residual)
        return self.relu(out)


class ResidualBlock(nn.Module):
    """Residual block for 1D convolutional neural networks."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)

        # Define a 1x1 convolutional layer to change the dimensions for shortcut connection, if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
            )

    def forward(self, x):
        out = nn.ReLU(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)  # Add shortcut connection
        out = nn.ReLU(out)
        return out


class MinPool1d(nn.Module):
    """Min pooling layer."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool1d(*args, **kwargs)

    def forward(self, x):
        return -self.pool(-x)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        L, D = x.size(1), x.size(-1)  # L: sequence length, D: embedding dimension
        device = x.device
        pos = torch.arange(0, L, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, D, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / D))
        pos_enc = torch.zeros((L, D), device=device)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        return x + pos_enc


class HierarchicalPositionalEncoding(nn.Module):
    """Hierarchical positional encoding for transformer models."""

    def __init__(self, d_model, num_time_steps, num_tokens_per_step):
        super(HierarchicalPositionalEncoding, self).__init__()
        self.d_model = d_model

        # Time-Step Level Encoding
        self.time_step_encoding = torch.randn(1, num_time_steps, d_model) / torch.sqrt(torch.tensor(d_model).float())

        # Token Level Encoding
        position = torch.arange(0, num_tokens_per_step).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.token_pos_enc = torch.zeros((1, num_tokens_per_step, d_model))
        self.token_pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        self.token_pos_enc[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        # x shape: [batch_size, num_time_steps * num_tokens_per_step, d_model]
        B, L, D = x.size()
        T, P = self.time_step_encoding.size(1), self.token_pos_enc.size(1)

        # Expand time-step encoding to match x's length
        time_step_enc_expanded = self.time_step_encoding.repeat(1, P, 1).expand(B, T * P, D)
        token_pos_enc_expanded = self.token_pos_enc.repeat(1, T, 1)

        # Sum the two encodings
        pos_enc = time_step_enc_expanded + token_pos_enc_expanded

        return x + pos_enc.to(x.device)


class PositionEncoding1dCNN:
    def __init__(self):
        super(PositionEncoding1dCNN, self).__init__()


def sinusoidal_positional_encoding(position, d_model):
    """Sinusoidal positional encoding for transformer models."""
    position = torch.arange(position).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

    pos_embedding = torch.zeros((position.size(0), d_model))
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)

    # center around 0.5
    pos_embedding = 0.5 * pos_embedding + 0.5
    if d_model == 1:
        pos_embedding = pos_embedding.squeeze(-1)

    return pos_embedding
