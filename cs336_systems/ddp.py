# cs336_systems/ddp.py
import torch
import torch.distributed as dist


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._handles = []

        # Broadcast parameters from rank 0
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Register backward hooks — use a set to avoid duplicate hooks
        # for tied (shared) parameters
        seen = set()
        for param in self.module.parameters():
            if param.requires_grad and param.data_ptr() not in seen:
                seen.add(param.data_ptr())
                param.register_post_accumulate_grad_hook(self._make_hook())

    def _make_hook(self):
        def hook(param):
            handle = dist.all_reduce(
                param.grad, op=dist.ReduceOp.SUM, async_op=True
            )
            self._handles.append((param, handle))
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for param, handle in self._handles:
            handle.wait()
            param.grad.div_(world_size)
        self._handles.clear()


class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float | None):
        super().__init__()
        self.module = module

        # Broadcast parameters from rank 0
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Build buckets in reverse parameter order (matching backward pass order)
        self._buckets = []  # list of lists of params
        self._param_to_bucket = {}  # param -> bucket index

        bucket_size_bytes = float("inf") if bucket_size_mb is None else bucket_size_mb * 1024 * 1024
        current_bucket = []
        current_size = 0

        # Reverse order: last parameters get gradients first in backward
        grad_params = []
        seen = set()
        for param in module.parameters():
            if param.requires_grad and param.data_ptr() not in seen:
                seen.add(param.data_ptr())
                grad_params.append(param)

        for param in reversed(grad_params):
            param_size = param.numel() * param.element_size()
            if current_bucket and current_size + param_size > bucket_size_bytes:
                self._buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            current_bucket.append(param)
            current_size += param_size

        if current_bucket:
            self._buckets.append(current_bucket)

        # Map each param to its bucket index
        for bucket_idx, bucket in enumerate(self._buckets):
            for param in bucket:
                self._param_to_bucket[param.data_ptr()] = bucket_idx

        # Track how many grads are ready per bucket
        self._bucket_pending = [len(b) for b in self._buckets]
        self._handles = {}  # bucket_idx -> handle

        # Register hooks
        for param in grad_params:
            param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param):
        bucket_idx = self._param_to_bucket[param.data_ptr()]

        def hook(p):
            self._bucket_pending[bucket_idx] -= 1
            if self._bucket_pending[bucket_idx] == 0:
                # All grads in this bucket are ready, launch all-reduce
                bucket_params = self._buckets[bucket_idx]
                # Flatten all grads in bucket into one tensor
                flat_grads = torch.cat([bp.grad.flatten() for bp in bucket_params])
                handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                self._handles[bucket_idx] = (handle, flat_grads)

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for bucket_idx, bucket in enumerate(self._buckets):
            if bucket_idx in self._handles:
                handle, flat_grads = self._handles[bucket_idx]
                handle.wait()
                flat_grads.div_(world_size)
                # Copy back to individual param grads
                offset = 0
                for param in bucket:
                    numel = param.numel()
                    param.grad.copy_(flat_grads[offset:offset + numel].view_as(param.grad))
                    offset += numel

        self._handles.clear()

    def reset_buckets(self):
        self._bucket_pending = [len(b) for b in self._buckets]
        self._handles.clear()
