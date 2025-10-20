from __future__ import annotations

from typing import Any, Dict, Optional, Union, Iterable
import json
import io
import numpy as np
import pandas as pd
import requests
from pprint import pprint

from uncertainty_engine_types import ResourceID
from uncertainty_engine import Client

def get_presigned_url(url: str) -> requests.Response:
    """
    Fetch the contents of a (pre-signed) URL.

    This function replaces the scheme 'https://' with 'http://' prior to the request,
    mirroring the behavior in the original code (some pre-signed endpoints may require it).

    Parameters
    ----------
    url : str
        The original (likely pre-signed) URL.

    Returns
    -------
    requests.Response
        The HTTP response object. Use `.content` / `.text` to access the payload.

    Raises
    ------
    requests.HTTPError
        If the response indicates an HTTP error status.
    """
    url = url.replace("https://", "http://")
    response = requests.get(url)
    response.raise_for_status()
    return response


def wrap_resource_id(resource_id: str) -> Dict[str, Any]:
    """
    Wrap a raw resource ID string into the `ResourceID` pydantic model's dict representation.

    Parameters
    ----------
    resource_id : str
        Raw resource ID.

    Returns
    -------
    dict
        A dictionary produced by `ResourceID(id=resource_id).model_dump()`.
    """
    return ResourceID(id=resource_id).model_dump()


def get_project_id(client: Client, project_name: str) -> str:
    """
    Resolve a project name to its unique project ID.

    Parameters
    ----------
    client : uncertainty_engine.Client
        Initialized API client.
    project_name : str
        Human-readable project name.

    Returns
    -------
    str
        The unique project ID.

    Raises
    ------
    ValueError
        If the project name does not exist.
    """
    projects = client.projects.list_projects()
    for project in projects:
        if project.name == project_name:
            return project.id
    raise ValueError(f"Project with name {project_name!r} not found.")


def get_workflow_id(client: Client, project_name: str, workflow_name: str) -> str:
    """
    Resolve a workflow name (within a project) to its unique workflow ID.

    Parameters
    ----------
    client : uncertainty_engine.Client
        Initialized API client.
    project_name : str
        Project name containing the workflow.
    workflow_name : str
        Human-readable workflow name.

    Returns
    -------
    str
        The unique workflow ID.

    Raises
    ------
    ValueError
        If the workflow name does not exist in the given project.
    """
    project_id = get_project_id(client, project_name)
    workflows = client.workflows.list_workflows(project_id)
    for workflow in workflows:
        if workflow.name == workflow_name:
            return workflow.id
    raise ValueError(f"Workflow with name {workflow_name!r} not found in project {project_name!r}.")


def get_resource_id(client: Client, project_name: str, resource_name: str, resource_type: str) -> str:
    """
    Resolve a resource name (within a project) to its unique resource ID.

    Parameters
    ----------
    client : uncertainty_engine.Client
        Initialized API client.
    project_name : str
        Project name containing the resource.
    resource_name : str
        Human-readable resource name.
    resource_type : str
        Resource type (e.g., "dataset", "model", etc.).

    Returns
    -------
    str
        The unique resource ID.

    Raises
    ------
    ValueError
        If the resource name does not exist in the given project.
    """
    project_id = get_project_id(client, project_name)
    resources = client.resources.list_resources(project_id, resource_type=resource_type)
    for resource in resources:
        if resource.name == resource_name:
            return resource.id
    raise ValueError(
        f"Resource with name {resource_name!r} (type={resource_type!r}) not found in project {project_name!r}."
    )


def upload_dataset(
    client: Client,
    project_name: str,
    dataset_name: str,
    file_path: Optional[str] = None,
    dataset: Optional[Union[pd.DataFrame, Dict[str, Iterable[Any]]]] = None,
    is_replace: bool = True,
    is_verbose: bool = False,
) -> None:
    """
    Upload a dataset to a project, from either a CSV file on disk or an in-memory dataset.

    Behavior:
    - If `file_path` is provided, that file is uploaded.
    - Else, if `dataset` is provided, it is written to `{dataset_name}.csv` and uploaded.
    - If the upload fails and `is_replace=True`, an update call is attempted instead.
    - Prints a confirmation upon success.

    Parameters
    ----------
    client : uncertainty_engine.Client
        Initialized API client.
    project_name : str
        Target project name.
    dataset_name : str
        Name to assign to the uploaded dataset resource.
    file_path : str, optional
        Path to a CSV file to upload.
    dataset : pandas.DataFrame or dict-like, optional
        In-memory dataset to upload. If dict-like, it will be converted to a DataFrame.
        Only used if `file_path` is None.
    is_replace : bool, default True
        If True, attempt to replace/update the existing dataset on error.

    Raises
    ------
    ValueError
        If neither `file_path` nor `dataset` is provided.
    """
    project_id = get_project_id(client, project_name)

    # Prepare a CSV on disk if only an in-memory dataset is provided.
    if file_path is None and dataset is not None:
        file_path = f"{dataset_name}.csv"
        df = dataset if isinstance(dataset, pd.DataFrame) else pd.DataFrame(dataset)
        df.to_csv(file_path, index=False)
    elif file_path is None and dataset is None:
        raise ValueError("Either `file_path` or `dataset` must be provided.")

    try:
        client.resources.upload(
            project_id=project_id,
            name=dataset_name,
            resource_type="dataset",
            file_path=file_path,
        )
    except Exception as e:
        if is_replace:
            client.resources.update(
                project_id=project_id,
                resource_id=get_resource_id(client, project_name, dataset_name, resource_type="dataset"),
                resource_type="dataset",
                file_path=file_path,
            )
        else:
            print(f"Error uploading dataset: {e}")
            return
    if is_verbose:
        print(f"Uploaded {dataset_name!r} to project {project_name!r}.")


def get_node_info(client: Client, node_name: str) -> None:
    """
    Print details for a specific node by its ID/name, using the client's `list_nodes()`.

    Parameters
    ----------
    client : uncertainty_engine.Client
        Initialized API client.
    node_name : str
        Node identifier to look up (as returned by `list_nodes()`).

    Returns
    -------
    None
        This function prints the node info to stdout.
    """
    nodes = client.list_nodes()
    nodes_by_id = {node["id"]: node for node in nodes}
    pprint(nodes_by_id[node_name])


def get_resource(client,
             project_name: str,
             resource_name: str,
             resource_type: str):
    """
    Download a resource from the Uncertainty Engine.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param resource_name: The name of the dataset.
    :return: The dataset as a pandas DataFrame.
    """
    response: bytes = client.resources.download(
        project_id=get_project_id(client=client, project_name=project_name),
        resource_type=resource_type,
        resource_id=get_resource_id(
            client=client,
            project_name=project_name,
            resource_name=resource_name,
            resource_type=resource_type,
        ),
    )
    decoded = response.decode("utf-8")
    return decoded


def get_data(client: Client, project_name: str, dataset_name: str) -> pd.DataFrame:
    """
    Download a named dataset from a project and return it as a pandas DataFrame.

    Parameters
    ----------
    client : uncertainty_engine.Client
        Initialized API client.
    project_name : str
        Name of the project containing the dataset.
    dataset_name : str
        Name of the dataset resource to download.

    Returns
    -------
    pandas.DataFrame
        Parsed dataset as a DataFrame.

    Raises
    ------
    ValueError
        If the project or dataset cannot be resolved to IDs.
    """
    decoded = get_resource(
        client=client,
        project_name=project_name,
        resource_name=dataset_name,
        resource_type='dataset'
    )
    df = pd.read_csv(io.StringIO(decoded))
    return df

def get_model(client,
                project_name: str,
                model_name: str):
    
    decoded = get_resource(
        client=client,
        project_name=project_name,
        resource_name=model_name,
        resource_type='model'
    )
    model = json.loads(decoded)
    return model

def get_model_inputs(client,
                     project_name: str,
                     model_name: str) -> Iterable[str]:
    """
    Get the input feature names for a model.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param model_name: The name of the model.
    :return: A list of input feature names.
    """
    model = get_model(
        client=client,
        project_name=project_name,
        model_name=model_name
    )
    input_features = model['metadata']['inputs']
    return input_features
