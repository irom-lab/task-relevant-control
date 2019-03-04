from .problem import ControlProblem, DSCProblem
from .policies import Policy
from .discrete_policies import DiscretePolicy, DiscreteTRVPolicy

import typing
import numpy as np

StateType = typing.Union[int, np.ndarray]
InputType = typing.Union[int, np.ndarray]
