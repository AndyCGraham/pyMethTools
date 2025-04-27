"""
Type annotations for improved static type checking with mypy.

This module provides TypedDicts and other custom type definitions for use in
pyMethTools to improve static type checking and IDE support.
"""

from typing import TypedDict, List, Dict, Tuple, Union, Optional, Callable, Any, TypeVar, Protocol
import numpy as np
import pandas as pd

# Type variables for generic functions
T = TypeVar('T')
U = TypeVar('U')

# Common types used across the codebase
RegionID = Union[int, float, str]
CpGIndex = int
RegionIndices = List[int]
MethylationCount = np.ndarray
CoverageCount = np.ndarray
DesignMatrix = Union[pd.DataFrame, np.ndarray]
ParameterArray = np.ndarray
AdjustmentFactor = Union[float, List[float], np.ndarray]

# TypedDicts for structured return types
class FitResult(TypedDict):
    params: np.ndarray
    standard_errors: np.ndarray
    
class SimulationResult(TypedDict):
    methylation: np.ndarray
    coverage: np.ndarray
    adjustments: Optional[np.ndarray]
    modified_regions: Optional[List]

class DMRResult(TypedDict):
    chr: str
    start: int
    end: int
    num_cpgs: int
    num_sig_cpgs: int
    prop_sig_cpgs: float

# Protocol for functions that can compute log-likelihoods
class LogLikelihoodFunction(Protocol):
    def __call__(self,
                count: np.ndarray,
                total: np.ndarray,
                X: np.ndarray,
                X_star: np.ndarray,
                beta: np.ndarray,
                n_param_abd: int,
                n_param_disp: int,
                link: str = ...,
                max_param: float = ...) -> float: ...

# Protocol for simulation functions
class SimulationFunction(Protocol):
    def __call__(self,
                params: np.ndarray,
                X: Union[pd.DataFrame, np.ndarray],
                X_star: Union[pd.DataFrame, np.ndarray],
                read_depth: int = ...,
                vary_read_depth: bool = ...,
                read_depth_sd: float = ...,
                adjust_factors: AdjustmentFactor = ...,
                sample_size: int = ...,
                link: str = ...) -> Tuple[np.ndarray, np.ndarray]: ...
