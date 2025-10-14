from __future__ import annotations
import requests
import io
import pandas as pd
from scipy.stats import norm
from pprint import pprint
import numpy as np

from uncertainty_engine_types import ResourceID
from uncertainty_engine import Client

def percentage_to_zscore(confidence_percent: float) -> float:
    """
    Convert a two-tailed confidence percentage to the corresponding z-score.
    
    Example:
        68% -> 1.00
        90% -> 1.645
        95% -> 1.96
        99% -> 2.576
    """
    confidence_fraction = confidence_percent / 100.0
    return norm.ppf(0.5 + confidence_fraction / 2.0)


def slice_dataframe(df, freeze_param, value):
    """
    Extract a slice from a multidimensional (meshgrid-flattened) DataFrame
    by fixing one parameter at a specified value.

    This is typically used to reduce an N-dimensional parameter sweep
    (stored as a flattened DataFrame) down to an (N–1)-dimensional slice
    by holding one variable constant and allowing all others to vary.

    The function identifies the unique grid values for the chosen parameter,
    applies a small fractional offset (+0.1%) to the requested value to
    mitigate floating-point rounding issues, and uses a tolerance based on
    half the minimum grid spacing to select matching rows.

    Parameters
    ----------
    df : pandas.DataFrame
        The full meshgrid-flattened DataFrame (e.g. all combinations of parameters).
    freeze_param : str
        The name of the column (parameter) to hold fixed.
    value : float
        The target value for the frozen parameter.

    Returns
    -------
    sliced : pandas.DataFrame
        A subset of `df` containing only the rows where the specified
        parameter equals (within tolerance) the requested value.
        The index is reset.

    Notes
    -----
    - If the target value does not exactly match any grid point due to
      floating-point precision, the small delta adjustment (1e-3)
      increases robustness of the match.
    - The tolerance is computed as half of the minimum grid spacing,
      assuming uniform spacing in the parameter grid.

    Examples
    --------
    >>> sliced = slice_dataframe(df, 'li6_prop', 0.25)
    >>> print(sliced.head())
    >>> print(sliced.shape)
    """
    u = np.unique(df[freeze_param])
    delta = 1.e-3
    value *= (1. + delta)
    tol = 0.5 * np.min(np.diff(u)) if len(u) > 1 else 1.e-12
    sliced = df[np.isclose(df[freeze_param], value, atol=tol)].reset_index(drop=True)
    return sliced

def get_presigned_url(url):
    """
    Get the contents from the presigned url.
    """
    url = url.replace("https://", "http://")
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response

def wrap_resource_id(resource_id: str):
  return ResourceID(id=resource_id).model_dump()

def get_project_id(client: Client, project_name: str) -> str:
    """
    Get the project ID from the project name.
    """
    projects = client.projects.list_projects()
    for project in projects:
        if project.name == project_name:
            return project.id
    raise ValueError(f"Project with name {project_name} not found.")

def get_workflow_id(client: Client, project_name: str, workflow_name: str) -> str:
    """
    Get the workflow ID from the workflow name.
    """
    project_id = get_project_id(client, project_name)
    workflows = client.workflows.list_workflows(project_id)
    for workflow in workflows:
        if workflow.name == workflow_name:
            return workflow.id
    raise ValueError(f"Workflow with name {workflow_name} not found.")

def get_resource_id(client, project_name: str, resource_name: str, resource_type: str) -> str:
    """
    Get the resource ID from the workflow name.
    """
    project_id = get_project_id(client, project_name)
    resources = client.resources.list_resources(project_id, resource_type=resource_type)
    for resource in resources:
        if resource.name == resource_name:
            return resource.id
    raise ValueError(f"Resource with name {resource_name} not found.")

def upload_dataset(client, project_name, dataset_name, file_path=None, dataset=None, is_replace=True):
    PROJECT_ID = get_project_id(client, project_name)
    if file_path is None and dataset is not None:

        file_path = f'{dataset_name}.csv'

        df = pd.DataFrame(dataset)
        df.to_csv(file_path, index=False)
    elif file_path is None and dataset is None:
        raise ValueError("Either file_path or dataset must be provided.")
    PROJECT_ID = get_project_id(client, project_name)
    try:
        client.resources.upload(
            project_id=PROJECT_ID,
            name=dataset_name,
            resource_type="dataset",
            file_path=file_path,
        )
    except Exception as e:
        if is_replace:
            client.resources.update(
                project_id=PROJECT_ID,
                resource_id=get_resource_id(client, project_name, dataset_name, resource_type="dataset"),
                resource_type="dataset",
                file_path=file_path,
            )
        else:
            print(f"Error uploading dataset: {e}")
    print(f'Uploaded {dataset_name} to {project_name}')

def get_node_info(client, node_name):
    nodes = client.list_nodes()
    nodes_by_id = {node["id"]: node for node in nodes}
    # Print the details of the node
    pprint(nodes_by_id[node_name])

def get_data(client,
             project_name: str,
             dataset_name: str,):
    response = client.resources.download(
            project_id=get_project_id(client=client, project_name=project_name),
            resource_type='dataset',
            resource_id=get_resource_id(client=client, project_name=project_name,
                                resource_name=dataset_name, resource_type='dataset')
        )
    decoded = response.decode("utf-8")
    df = pd.read_csv(io.StringIO(decoded))
    return df

