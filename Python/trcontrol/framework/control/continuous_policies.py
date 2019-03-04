from typing import Union

import numpy as np

from .policies import Policy


class LQRPolicy(Policy):
    def input(self, state: Union[int, np.ndarray], t: int):
        pass

    def solve(self):
        pass