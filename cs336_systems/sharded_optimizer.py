from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    """Shard optimizer state across ranks and broadcast updated parameters.

    This is a ZeRO-1 style wrapper: every rank keeps gradients for the whole
    model, but each rank only owns and optimizer-steps a subset of parameters.
    After the local optimizer step, owners broadcast their updated parameters so
    all ranks keep identical model weights.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ):
        param_groups = self._materialize_param_groups(params)

        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self._all_params = self._deduplicate_params(param_groups)
        self._param_to_owner = {id(param): idx % self._world_size for idx, param in enumerate(self._all_params)}

        local_param_groups: list[dict[str, Any]] = []
        self._local_group_to_global_group: list[int] = []
        for group_idx, group in enumerate(param_groups):
            local_params = [param for param in group["params"] if self._param_to_owner[id(param)] == self._rank]
            if not local_params:
                continue

            local_group = {key: value for key, value in group.items() if key != "params"}
            local_group["params"] = local_params
            local_param_groups.append(local_group)
            self._local_group_to_global_group.append(group_idx)

        self._local_optimizer = optimizer_cls(local_param_groups, **kwargs) if local_param_groups else None
        defaults = self._local_optimizer.defaults if self._local_optimizer is not None else kwargs
        super().__init__(param_groups, defaults)

    @staticmethod
    def _materialize_param_groups(
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")

        if isinstance(params[0], dict):
            param_groups = []
            for group in params:
                materialized_group = dict(group)
                materialized_group["params"] = list(materialized_group["params"])
                param_groups.append(materialized_group)
            return param_groups

        return [{"params": params}]

    @staticmethod
    def _deduplicate_params(param_groups: list[dict[str, Any]]) -> list[torch.Tensor]:
        params = []
        seen = set()
        for group in param_groups:
            for param in group["params"]:
                if id(param) in seen:
                    continue
                seen.add(id(param))
                params.append(param)
        return params

    def _sync_local_group_options(self) -> None:
        if self._local_optimizer is None:
            return

        for local_group, global_group_idx in zip(
            self._local_optimizer.param_groups,
            self._local_group_to_global_group,
            strict=True,
        ):
            global_group = self.param_groups[global_group_idx]
            local_params = local_group["params"]
            local_group.clear()
            local_group.update({key: value for key, value in global_group.items() if key != "params"})
            local_group["params"] = local_params

    def _broadcast_updated_params(self) -> None:
        if self._world_size == 1:
            return

        for param in self._all_params:
            dist.broadcast(param.data, src=self._param_to_owner[id(param)])

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    param.grad.zero_()

    def step(self, closure=None):  # type: ignore[override]
        self._sync_local_group_options()

        loss = None
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure=closure)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._broadcast_updated_params()
        return loss

    def state_dict(self):  # type: ignore[override]
        if self._local_optimizer is None:
            return {"state": {}, "param_groups": []}
        return self._local_optimizer.state_dict()

    def load_state_dict(self, state_dict):  # type: ignore[override]
        if self._local_optimizer is None:
            return None
        return self._local_optimizer.load_state_dict(state_dict)
