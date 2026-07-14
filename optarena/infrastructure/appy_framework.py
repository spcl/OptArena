# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later

from optarena.infrastructure import Framework
from optarena.infrastructure.framework import TorchCudaEventTiming
from typing import Any, Callable, Dict


class APPyFramework(TorchCudaEventTiming, Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return 0.1

    def copy_func(self) -> Callable:
        import torch
        torch.set_default_device('cuda')

        def inner(arr):
            copy = torch.from_numpy(arr).to('cuda')
            return copy

        return inner

    def imports(self) -> Dict[str, Any]:
        import torch
        return {'torch': torch}

    def post_call(self, result: Any) -> Any:
        """Sync the CUDA stream so timing captures the async kernel (replaces
        the ``; torch.cuda.synchronize()`` appended to an exec string)."""
        import torch
        torch.cuda.synchronize()
        return result

    # Native GPU timing (torch CUDA events) comes from the TorchCudaEventTiming
    # mixin -- shared verbatim with Triton.
