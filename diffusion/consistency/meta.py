import torch as th


class MetaConsistency:
    def __init__(
            self,
            sigma_max=1000,
            sigma_min=0,
            weight_schedule="uniform",
            boundary_func='truncate'
    ):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.boundary_func = boundary_func

    def get_scalings_for_boundary_condition(self, sigma):
        if self.boundary_func == 'sigmoid':
            c_out = th.sigmoid(12 * (sigma - self.sigma_max / 2) / self.sigma_max) * sigma / (
                        sigma + 1e-8)  # sigmoid(-6, 6)
            c_skip = 1 - c_out
        elif self.boundary_func == 'linear':
            # c_out, c_skip = 0.5 + torch.zeros_like(sigma, device=sigma.device), 0.5 + torch.zeros_like(sigma, device=sigma.device)
            c_out = sigma / self.sigma_max
            c_skip = 1 - c_out
        elif self.boundary_func == 'truncate':
            c_out = sigma / (sigma + 1e-9)
            c_skip = 1 - c_out
        else:
            raise NotImplementedError
        return c_skip, c_out

    def get_weightings(self, weight_schedule, snrs, sigma_data):
        if weight_schedule == "snr":
            weightings = snrs
        elif weight_schedule == "snr+1":
            weightings = snrs + 1
        elif weight_schedule == "karras":
            weightings = snrs + 1.0 / sigma_data ** 2
        elif weight_schedule == "truncated-snr":
            weightings = th.clamp(snrs, min=1.0)
        elif weight_schedule == "uniform":
            weightings = th.ones_like(snrs)
        else:
            raise NotImplementedError()
        return weightings

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    def consistency_losses(
            self,
            model,
            batch
    ):
        pass

    def consistency_test_step(
            self,
            model,
            batch,
            batch_idx,
            split='test'
    ):
        pass
