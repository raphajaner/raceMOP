import numpy as np
from torchinfo import summary
import flatdict
from omegaconf import OmegaConf

import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Layer initialization as in the PPO paper"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_xavier(layer, std=np.sqrt(2), bias_const=0.0):
    """Layer initialization as in the PPO paper"""
    nn.init.xavier_uniform_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_kaiming(layer, std=np.sqrt(2), bias_const=0.0):
    """Layer initialization as in the PPO paper"""
    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    nn.init.constant_(layer.bias, bias_const)
    return layer


def log_hyperparams(writer, config):
    """Log hyperparameters"""
    dict_config = flatdict.FlatDict(OmegaConf.to_container(config))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in dict_config.items()])),
    )


def conv1d_out_dim(size, conv_arch):
    """Computes the output size of a convolutional network.

    Args:
        size (int): The input size.
        conv_arch (list): The convolutional architecture.
    Returns:
        int: The output size.
    """
    for _, _, kernel_size, stride, padding in conv_arch:
        size = (size - kernel_size + 2 * padding) // stride + 1
    return size * conv_arch[-1][1]


def calculate_output_size_1d_conv(W, K, P=0, S=1):
    """Calculate the output size of a 1D convolutional layer.

    Args:
        W (int): The input size.
        K (int): The kernel size.
        P (int): The padding.
        S (int): The stride.
    Returns:
        int: The output size.
    """
    O = (W - K + 2 * P) // S + 1
    return O


def save_model_summary(log_dir, models, input_data=None, verbose=0):
    """Save model summary to file.

    Args:
        config (DictConfig): Configuration dictionary.
        models (List[nn.Module]): List of models.
    """
    with open(f'{log_dir}/model_summary.txt', 'w') as file:
        for model in models:
            model_summary = summary(
                model,
                col_names=[
                    "num_params",
                    "params_percent",
                    "kernel_size",
                    "input_size",
                    "output_size",
                    # "mult_adds", "trainable",
                ],
                input_data=[input_data[0].to_dict()],
                row_settings=("var_names", "depth"),
                depth=10,
                verbose=verbose
            )
            try:
                file.write(repr(model_summary) + '\n')
            except:
                pass


def conv1d_output_size(W, K, P=0, S=1):
    """Calculate the output size of a 1D convolutional layer.

    Args:
        W (int): The input size.
        K (int): The kernel size.
        P (int): The padding.
        S (int): The stride.
    Returns:
        int: The output size.
    """
    O = (W - K + 2 * P) // S + 1
    return O


def conv1d_factory(conv_arch, in_channels, out_channels):
    """Create a 1D convolutional layer.

    Args:
        conv_arch (list): The convolutional architecture.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        nn.Sequential: The convolutional layer.
    """
    layers = []
    for kernel_size, stride, padding in conv_arch:
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
        layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.Sequential(*layers)


def linear_factory(linear_arch, in_features, out_features):
    """Create a linear layer.

    Args:
        linear_arch (list): The linear architecture.
        in_features (int): The number of input features.
        out_features (int): The number of output features.
    Returns:
        nn.Sequential: The linear layer.
    """
    layers = []
    for out_features in linear_arch:
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    return nn.Sequential(*layers)
