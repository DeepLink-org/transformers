import torch

from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.distributed.fsdp._runtime_utils import (
    _get_fsdp_root_states,
    _is_fsdp_root,
    _lazy_init,
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_unshard,
    _root_pre_forward,
)
import functools
from torch.distributed.fsdp._utils import p_assert

class my_FullyShardedDataParallel(FullyShardedDataParallel):
  def generate(self, *args, **kwargs):
      """
      Runs the forward pass for the wrapped module, inserting FSDP-specific
      pre- and post-forward sharding logic.
      """
      with torch.autograd.profiler.record_function(
          "FullyShardedDataParallel.generate"
      ):
          args, kwargs = _root_pre_forward(self, self, args, kwargs)
          unused = None
          unshard_fn = functools.partial(_pre_forward_unshard, self, self._handles)
          reshard_fn = functools.partial(_post_forward_reshard, self, self._handles)
          args, kwargs = _pre_forward(
              self, self._handles, unshard_fn, self._fsdp_wrapped_module, args, kwargs
          )
          for handle in self._handles:
              p_assert(
                  handle.flat_param.device == self.compute_device,
                  "Expected `FlatParameter` to be on the compute device "
                  f"{self.compute_device} but got {handle.flat_param.device}",
              )
          output = self._fsdp_wrapped_module.generate(*args, **kwargs)
          return _post_forward(self, self._handles, reshard_fn, self, unused, output)

torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel = my_FullyShardedDataParallel