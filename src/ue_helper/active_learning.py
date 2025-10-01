"""
Active Learning Helper Functions

This module provides clean, reusable helper functions for active learning
extracted from the Branin function optimization workbook.
"""

import numpy as np
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from uncertainty_engine import Client
from uncertainty_engine.graph import Graph
from uncertainty_engine.nodes.base import Node
from uncertainty_engine_types import ResourceID
from uncertainty_engine.nodes.workflow import Workflow


def active_learning_step(client, project_id, inputs, outputs, acquisition_function="PosteriorStandardDeviation", num_points=1):
    """
    Perform one step of active learning using Uncertainty Engine.
    
    This function takes current data, trains a model, and recommends new points
    to evaluate based on the specified acquisition function.
    
    Parameters:
    -----------
    client : uncertainty_engine.Client
        Authenticated Uncertainty Engine client
    project_id : str
        Project ID where datasets will be uploaded
    inputs : list of lists
        Input coordinates (e.g., [[x1, y1], [x2, y2], ...])
    outputs : list of lists
        Output values (e.g., [[z1], [z2], ...])
    acquisition_function : str, optional
        Acquisition function to use (default: "PosteriorStandardDeviation")
    num_points : int, optional
        Number of points to recommend (default: 1)
    
    Returns:
    --------
    tuple : (recommended_points_df, model_output, dataset_names)
        - recommended_points_df: DataFrame with recommended points
        - model_output: The trained model output from the workflow
        - dataset_names: Dictionary with 'inputs_name' and 'outputs_name' keys for cleanup
    
    Example:
    --------
    >>> # Initialize client and authenticate
    >>> client = Client()
    >>> client.authenticate("your_account_id")
    >>> 
    >>> # Prepare your data
    >>> inputs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    >>> outputs = [[10.5], [15.2], [8.7]]
    >>> 
    >>> # Get recommendations
    >>> recommendations, model, dataset_names = active_learning_step(
    ...     client=client,
    ...     project_id="your_project_id",
    ...     inputs=inputs,
    ...     outputs=outputs,
    ...     acquisition_function="PosteriorStandardDeviation",
    ...     num_points=2
    ... )
    >>> 
    >>> print("Recommended points:")
    >>> print(recommendations)
    """
    
    # Generate timestamp for unique file names
    timestamp = datetime.now().strftime("%d%m%y%H%M%S")
    
    # Create temporary CSV files
    csv_filename_inputs = f"temp_inputs_{timestamp}.csv"
    csv_filename_outputs = f"temp_outputs_{timestamp}.csv"
    
    try:
        # Save inputs data
        with open(csv_filename_inputs, 'w') as f:
            for input_point in inputs:
                f.write(",".join(map(str, input_point)) + "\n")
        
        # Save outputs data
        with open(csv_filename_outputs, 'w') as f:
            for output_point in outputs:
                f.write(",".join(map(str, output_point)) + "\n")
        
        # Upload datasets
        client.resources.upload(
            project_id=project_id,
            name=f"ActiveLearning_inputs_{timestamp}",
            file_path=csv_filename_inputs,
            resource_type="dataset"
        )
        
        client.resources.upload(
            project_id=project_id,
            name=f"ActiveLearning_outputs_{timestamp}",
            file_path=csv_filename_outputs,
            resource_type="dataset"
        )
        
        # Get resources dictionary
        resources_list = client.resources.list_resources(project_id=project_id, resource_type="dataset")
        resources_dict = {resource.name: resource.id for resource in resources_list}
        
        # Create workflow graph
        graph = Graph()
        
        # Load inputs dataset
        data_loader_inputs = Node(
            node_name="Load",
            label="Load Inputs Data",
            project_id=project_id,
            file_id=ResourceID(id=resources_dict[f"ActiveLearning_inputs_{timestamp}"]).model_dump(),
            file_type="dataset",
        )
        inputs_handle = data_loader_inputs.make_handle("file")
        graph.add_node(data_loader_inputs)
        
        # Load outputs dataset
        data_loader_outputs = Node(
            node_name="Load",
            label="Load Outputs Data",
            project_id=project_id,
            file_id=ResourceID(id=resources_dict[f"ActiveLearning_outputs_{timestamp}"]).model_dump(),
            file_type="dataset",
        )
        outputs_handle = data_loader_outputs.make_handle("file")
        graph.add_node(data_loader_outputs)
        
        # Model configuration
        model_config = Node(
            node_name="ModelConfig",
            label="Model Config",
        )
        config_handle = model_config.make_handle("config")
        graph.add_node(model_config)
        
        # Training node
        train_model = Node(
            node_name="TrainModel",
            label="Train Model",
            config=config_handle,
            inputs=inputs_handle,
            outputs=outputs_handle,
        )
        output_model = train_model.make_handle("model")
        graph.add_node(train_model)
        
        # Recommend node
        recommend_node = Node(
            node_name="Recommend",
            label="Recommend",
            model=output_model,
            acquisition_function=acquisition_function,
            number_of_points=num_points,
        )
        recommend_handle = recommend_node.make_handle("recommended_points")
        graph.add_node(recommend_node)
        
        # Download recommendations
        download_recommendations = Node(
            node_name="Download",
            label="Download Recommendations",
            file=recommend_handle,
        )
        download_handle = download_recommendations.make_handle("file")
        graph.add_node(download_recommendations)
        
        # Create and run workflow
        workflow = Workflow(
            graph=graph.nodes,
            input=graph.external_input,
            external_input_id=graph.external_input_id,
            requested_output={
                "trained_model": output_model.model_dump(),
                "recommended_points": download_handle.model_dump(),
            },
        )
        
        # Run the workflow
        job_response = client.run_node(workflow)
        
        # Get recommended points
        recommended_points_url = job_response.outputs["outputs"]["recommended_points"]
        
        # Fetch the CSV content
        response = requests.get(recommended_points_url)
        response.raise_for_status()
        
        # Read the recommended points
        recommended_points_df = pd.read_csv(StringIO(response.text))
        
        # Return dataset names for cleanup
        dataset_names = {
            'inputs_name': f"ActiveLearning_inputs_{timestamp}",
            'outputs_name': f"ActiveLearning_outputs_{timestamp}"
        }
        
        return recommended_points_df, job_response.outputs["outputs"]["trained_model"], dataset_names
        
    except Exception as e:
        raise Exception(f"Active learning step failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        import os
        for filename in [csv_filename_inputs, csv_filename_outputs]:
            if os.path.exists(filename):
                os.remove(filename)


def active_learning_loop(client, project_id, initial_inputs, initial_outputs, 
                        objective_function, num_iterations=5, 
                        acquisition_function="PosteriorStandardDeviation", 
                        num_points_per_iteration=1):
    """
    Run a complete active learning optimization loop.
    
    Parameters:
    -----------
    client : uncertainty_engine.Client
        Authenticated Uncertainty Engine client
    project_id : str
        Project ID where datasets will be uploaded
    initial_inputs : list of lists
        Initial input coordinates
    initial_outputs : list of lists
        Initial output values
    objective_function : callable
        Function to evaluate new points (takes a list of coordinates, returns a value)
    num_iterations : int, optional
        Number of active learning iterations (default: 5)
    acquisition_function : str, optional
        Acquisition function to use (default: "PosteriorStandardDeviation")
    num_points_per_iteration : int, optional
        Number of points to recommend per iteration (default: 1)
    
    Returns:
    --------
    dict : Results containing:
        - 'final_inputs': Final input coordinates
        - 'final_outputs': Final output values
        - 'best_point': Best point found (coordinates, value)
        - 'convergence_history': List of best values over iterations
        - 'all_recommendations': List of all recommended points
    
    Example:
    --------
    >>> def my_objective(x):
    ...     return x[0]**2 + x[1]**2  # Simple 2D quadratic
    >>> 
    >>> results = active_learning_loop(
    ...     client=client,
    ...     project_id="your_project_id",
    ...     initial_inputs=[[0.0, 0.0], [1.0, 1.0]],
    ...     initial_outputs=[[0.0], [2.0]],
    ...     objective_function=my_objective,
    ...     num_iterations=10
    ... )
    >>> 
    >>> print(f"Best point: {results['best_point']}")
    """
    
    # Initialize current data
    current_inputs = initial_inputs.copy()
    current_outputs = initial_outputs.copy()
    
    # Track results
    convergence_history = []
    all_recommendations = []
    
    print(f"Starting active learning loop with {num_iterations} iterations...")
    print(f"Initial data points: {len(current_inputs)}")
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        
        # Store the timestamp for this iteration's datasets
        iteration_timestamp = None
        
        try:
            # Get recommendations
            recommendations_df, model_output, dataset_names = active_learning_step(
                client=client,
                project_id=project_id,
                inputs=current_inputs,
                outputs=current_outputs,
                acquisition_function=acquisition_function,
                num_points=num_points_per_iteration
            )
            
            # Process recommendations
            if not recommendations_df.empty:
                for i, row in recommendations_df.iterrows():
                    # Extract coordinates (assuming first columns are input coordinates)
                    coords = list(row.values)
                    if len(coords) >= 2:
                        # For 2D case, assume first two columns are x, y
                        x_new, y_new = coords[0], coords[1]
                        
                        # Evaluate using objective function
                        output_value = objective_function([x_new, y_new])
                        
                        # Add to current data
                        current_inputs.append([x_new, y_new])
                        current_outputs.append([output_value])
                        
                        # Track recommendation
                        all_recommendations.append({
                            'iteration': iteration + 1,
                            'coordinates': [x_new, y_new],
                            'value': output_value
                        })
                        
                        print(f"  Added point: ({x_new:.4f}, {y_new:.4f}) -> {output_value:.4f}")
            
            # Update convergence history
            best_value = min([output[0] for output in current_outputs])
            convergence_history.append(best_value)
            
            print(f"  Current best value: {best_value:.4f}")
            print(f"  Total points: {len(current_inputs)}")
            
            # Clean up datasets from this iteration (except for the final iteration)
            if iteration < num_iterations - 1:  # Don't delete on the last iteration
                try:
                    # Get current resources to find the datasets created in this iteration
                    resources_list = client.resources.list_resources(project_id=project_id, resource_type="dataset")
                    resources_dict = {resource.name: resource.id for resource in resources_list}
                    
                    # Delete the specific datasets created in this iteration
                    for dataset_type, dataset_name in dataset_names.items():
                        if dataset_name in resources_dict:
                            try:
                                client.resources.delete_resource(
                                    project_id=project_id,
                                    resource_type="dataset",
                                    resource_id=resources_dict[dataset_name]
                                )
                                print(f"  🗑️ Deleted {dataset_type}: {dataset_name}")
                            except Exception as delete_error:
                                print(f"  ⚠️ Warning: Could not delete {dataset_name}: {delete_error}")
                    
                except Exception as cleanup_error:
                    print(f"  ⚠️ Warning: Could not clean up datasets for iteration {iteration + 1}: {cleanup_error}")
            
        except Exception as e:
            print(f"  Error in iteration {iteration + 1}: {e}")
            continue
    
    # Find best point
    best_idx = np.argmin([output[0] for output in current_outputs])
    best_point = {
        'coordinates': current_inputs[best_idx],
        'value': current_outputs[best_idx][0]
    }
    
    print(f"\n=== Active Learning Complete ===")
    print(f"Total evaluations: {len(current_inputs)}")
    print(f"Best point: {best_point['coordinates']} -> {best_point['value']:.6f}")
    
    return {
        'final_inputs': current_inputs,
        'final_outputs': current_outputs,
        'best_point': best_point,
        'convergence_history': convergence_history,
        'all_recommendations': all_recommendations
    }
