"""
Function Generators for Optimization Problems

This module provides generators for various optimization test functions
that can be used with the Uncertainty Engine active learning workflows.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class FunctionInfo:
    """Information about an optimization function."""
    name: str
    function: Callable
    search_space: List[Tuple[float, float]]
    global_minimum: float
    global_minimum_points: List[List[float]]
    description: str


def generate_branin_function() -> FunctionInfo:
    """
    Generate the Branin function for optimization.
    
    The Branin function is a 2D test function with 3 global minima.
    Domain: x1 in [-5, 10], x2 in [0, 15]
    Global minimum: f(x) ≈ 0.397887 at multiple points
    
    Returns:
    --------
    FunctionInfo : Object containing the function and metadata
    """
    
    def branin(x: List[float]) -> float:
        """
        Branin function: 2D test function with 3 global minima
        
        Parameters:
        -----------
        x : List[float]
            Input coordinates [x1, x2]
            
        Returns:
        --------
        float : Function value at the given point
        """
        x1, x2 = x
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        d = 6
        e = 10
        f = 1 / (8 * np.pi)
        
        term1 = a * (x2 - b * x1**2 + c * x1 - d)**2
        term2 = e * (1 - f) * np.cos(x1)
        term3 = e
        
        return term1 + term2 + term3
    
    # Define the search space
    search_space = [(-5.0, 10.0), (0.0, 15.0)]  # x1: [-5, 10], x2: [0, 15]
    
    # Global minimum points (approximate)
    global_minimum_points = [
        [-np.pi, 12.275],
        [np.pi, 2.275],
        [9.42478, 2.475]
    ]
    
    return FunctionInfo(
        name="Branin",
        function=branin,
        search_space=search_space,
        global_minimum=0.397887,
        global_minimum_points=global_minimum_points,
        description="2D test function with 3 global minima. Domain: x1 in [-5, 10], x2 in [0, 15]"
    )


def generate_ackley_function() -> FunctionInfo:
    """
    Generate the Ackley function for optimization.
    
    The Ackley function is a 2D test function with many local minima
    and one global minimum at (0, 0).
    
    Returns:
    --------
    FunctionInfo : Object containing the function and metadata
    """
    
    def ackley(x: List[float]) -> float:
        """
        Ackley function: 2D test function with many local minima
        
        Parameters:
        -----------
        x : List[float]
            Input coordinates [x1, x2]
            
        Returns:
        --------
        float : Function value at the given point
        """
        x1, x2 = x
        a = 20
        b = 0.2
        c = 2 * np.pi
        
        term1 = -a * np.exp(-b * np.sqrt(0.5 * (x1**2 + x2**2)))
        term2 = -np.exp(0.5 * (np.cos(c * x1) + np.cos(c * x2)))
        term3 = a + np.e
        
        return term1 + term2 + term3
    
    # Define the search space
    search_space = [(-5.0, 5.0), (-5.0, 5.0)]
    
    return FunctionInfo(
        name="Ackley",
        function=ackley,
        search_space=search_space,
        global_minimum=0.0,
        global_minimum_points=[[0.0, 0.0]],
        description="2D test function with many local minima and one global minimum at (0, 0)"
    )


def generate_rastrigin_function() -> FunctionInfo:
    """
    Generate the Rastrigin function for optimization.
    
    The Rastrigin function is a 2D test function with many local minima
    and one global minimum at (0, 0).
    
    Returns:
    --------
    FunctionInfo : Object containing the function and metadata
    """
    
    def rastrigin(x: List[float]) -> float:
        """
        Rastrigin function: 2D test function with many local minima
        
        Parameters:
        -----------
        x : List[float]
            Input coordinates [x1, x2]
            
        Returns:
        --------
        float : Function value at the given point
        """
        x1, x2 = x
        A = 10
        n = 2
        
        return A * n + (x1**2 - A * np.cos(2 * np.pi * x1)) + (x2**2 - A * np.cos(2 * np.pi * x2))
    
    # Define the search space
    search_space = [(-5.12, 5.12), (-5.12, 5.12)]
    
    return FunctionInfo(
        name="Rastrigin",
        function=rastrigin,
        search_space=search_space,
        global_minimum=0.0,
        global_minimum_points=[[0.0, 0.0]],
        description="2D test function with many local minima and one global minimum at (0, 0)"
    )


def generate_sphere_function() -> FunctionInfo:
    """
    Generate the Sphere function for optimization.
    
    The Sphere function is a simple 2D convex function
    with one global minimum at (0, 0).
    
    Returns:
    --------
    FunctionInfo : Object containing the function and metadata
    """
    
    def sphere(x: List[float]) -> float:
        """
        Sphere function: Simple 2D convex function
        
        Parameters:
        -----------
        x : List[float]
            Input coordinates [x1, x2]
            
        Returns:
        --------
        float : Function value at the given point
        """
        return x[0]**2 + x[1]**2
    
    # Define the search space
    search_space = [(-5.0, 5.0), (-5.0, 5.0)]
    
    return FunctionInfo(
        name="Sphere",
        function=sphere,
        search_space=search_space,
        global_minimum=0.0,
        global_minimum_points=[[0.0, 0.0]],
        description="Simple 2D convex function with one global minimum at (0, 0)"
    )


def get_function_generator(function_name: str) -> FunctionInfo:
    """
    Get a function generator by name.
    
    Parameters:
    -----------
    function_name : str
        Name of the function to generate. Options: 'branin', 'ackley', 'rastrigin', 'sphere'
        
    Returns:
    --------
    FunctionInfo : Object containing the function and metadata
        
    Raises:
    -------
    ValueError : If function_name is not recognized
    """
    generators = {
        'branin': generate_branin_function,
        'ackley': generate_ackley_function,
        'rastrigin': generate_rastrigin_function,
        'sphere': generate_sphere_function
    }
    
    if function_name.lower() not in generators:
        available = ', '.join(generators.keys())
        raise ValueError(f"Unknown function '{function_name}'. Available functions: {available}")
    
    return generators[function_name.lower()]()


def list_available_functions() -> List[str]:
    """
    List all available function generators.
    
    Returns:
    --------
    List[str] : List of available function names
    """
    return ['branin', 'ackley', 'rastrigin', 'sphere']


def latin_hypercube_sampling(space: List[Tuple[float, float]], n_samples: int, seed: int = None) -> np.ndarray:
    """
    Generate Latin Hypercube samples for better initial coverage.
    
    Parameters:
    -----------
    space : List[Tuple[float, float]]
        List of (min, max) tuples defining the search space
    n_samples : int
        Number of samples to generate
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray : Array of shape (n_samples, len(space)) containing the samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    bounds = np.array(space)
    samples = np.zeros((n_samples, len(space)))
    
    for i in range(len(space)):
        edges = np.linspace(bounds[i, 0], bounds[i, 1], n_samples + 1)
        segment = edges[1] - edges[0]
        samples[:, i] = edges[:-1] + np.random.random(n_samples) * segment
        np.random.shuffle(samples[:, i])
    
    return samples