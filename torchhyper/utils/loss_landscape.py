from typing import Dict

import torch


@torch.no_grad()
def filter_normalization(
        parameters_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    normalized_random_direction = {
        key: torch.randn_like(value)
        for key, value in parameters_dict.items()
    }

    for (param_name,
         param_data), rand_dir in zip(parameters_dict.items(),
                                      normalized_random_direction.values()):

        for i, (d, w) in enumerate(zip(rand_dir, param_data)):

            normalized_random_direction[param_name][i] = d * w.norm() / (
                d.norm() + 1e-8)

    return normalized_random_direction
