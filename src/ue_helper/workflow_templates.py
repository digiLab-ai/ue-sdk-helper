from __future__ import annotations
import numpy as np
from pprint import pprint
from typing import Any, Dict, Optional, Union, Iterable
import pandas as pd
from io import StringIO
from uncertainty_engine.nodes.base import Node
from uncertainty_engine.graph import Graph
from uncertainty_engine.nodes.workflow import Workflow

from ue_helper.utils import (
    get_project_id,
    get_resource_id,
    wrap_resource_id,
    get_presigned_url,
    get_model_inputs,
    get_data,
    upload_dataset,
)

def train_and_save_model_workflow(client, 
                   project_name: str,
                   save_model_name: str,
                   input_names: list,
                   output_names: list,
                   train_dataset: Optional[Union[str, pd.DataFrame, Dict[str, Iterable[Any]]]],
                   is_visualise_workflow: bool = False,
                   is_print_full_output: bool = False,
                   save_workflow_name: Union[str, None] = None
                   ) -> dict:
    """
    A workflow that trains a machine learning model.
    Here, we assume all resources have already been uploaded to the cloud.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param dataset_name: The name of the dataset.
    :param input_names: The names of the input columns.
    :param output_names: The names of the output columns.
    :param save_model_name: The name to save the trained model as.
    :param is_visualise_workflow: Whether to print the workflow graph.
    :param is_print_full_output: Whether to print the full output of the workflow.
    :return: The response from running the workflow.
    """

    train_dataset_name = '_train_data'
    is_upload_dataset = type(train_dataset) is not str
    if is_upload_dataset:
        upload_dataset(
            client=client,
            project_name=project_name,
            dataset_name=train_dataset_name,
            dataset=train_dataset
        )
    else:
        train_dataset_name = train_dataset
    
    train_dataset = get_data(
        client=client,
        project_name=project_name,
        dataset_name=train_dataset_name
    )
    
    # Check that the input names exist in the dataset
    missing = set(input_names) - set(train_dataset.columns)
    if missing:
        raise ValueError(
            f"The following input_names are missing from dataset columns: {sorted(missing)}\n"
            f"Available columns: {train_dataset.columns.tolist()}"
        )
    # Check that the output names exist in the dataset
    missing = set(output_names) - set(train_dataset.columns)
    if missing:
        raise ValueError(
            f"The following output_names are missing from dataset columns: {sorted(missing)}\n"
            f"Available columns: {train_dataset.columns.tolist()}"
        )
        

    # 1. Create the graph
    graph = Graph()

    # 2. Create relevant nodes, handles, and add to graph:

    # 2.a. Model config node
    model_config = Node(
        node_name="ModelConfig",
        label="Model Config",
    )
    graph.add_node(model_config)  # add to graph
    
    # 2.b. Load dataset node
    load_data = Node(
        node_name="LoadDataset",
        label="Load Dataset",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name, 
                resource_name=train_dataset_name, 
                resource_type='dataset')
                ),
        project_id=get_project_id(
            client=client,
            project_name=project_name
            ),
    )
    graph.add_node(load_data)  # add to graph
    
    # 2.b. Filter dataset node for inputs
    input_data = Node(
        node_name="FilterDataset",
        label="Input Dataset",
        columns=input_names,
        dataset=load_data.make_handle("file"),
    )
    graph.add_node(input_data)  # add to graph
    
    # 2.c. Filter dataset node for outputs
    output_data = Node(
        node_name="FilterDataset",
        label="Output Dataset",
        columns=output_names,
        dataset=load_data.make_handle("file"),
    )
    graph.add_node(output_data)  # add to graph
    
    # 2.d. Train model node
    train_model = Node(
        node_name="TrainModel",
        label="Train Model",
        config=model_config.make_handle("config"),
        inputs=input_data.make_handle("dataset"),
        outputs=output_data.make_handle("dataset"),
    )
    graph.add_node(train_model)  # add to graph
    
    # 2.e. Save model node
    save = Node(
        node_name="Save",
        label="Save",
        data=train_model.make_handle("model"),
        file_id=save_model_name,
        project_id=get_project_id(
            client=client,
            project_name=project_name
            ),
    )
    graph.add_node(save)  # add to graph

    if is_visualise_workflow:
        pprint(graph.nodes)

    workflow = Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        requested_output={
            }
        )
    if save_workflow_name is not None:
        client.workflows.save(
            project_id=get_project_id(
                client=client,
                project_name=project_name
                ),
            workflow=workflow,
            workflow_name=save_workflow_name
        )

    response = client.run_node(workflow)
    if is_print_full_output:
        pprint(response.model_dump())
        

def predict_model_workflow(client, 
                           predict_dataset: Optional[Union[str, pd.DataFrame, Dict[str, Iterable[Any]]]], 
                   project_name: str,
                   model_name: str,
                   input_names: Union[list, None] = None,
                   is_visualise_workflow: bool = False,
                   is_print_full_output: bool = False,
                   save_workflow_name: Union[str, None] = None) -> dict:
    """
    A workflow that trains a machine learning model.
    Here, we assume all resources have already been uploaded to the cloud.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param dataset_name: The name of the dataset.
    :param input_names: The names of the input columns.
    :param output_names: The names of the output columns.
    :param save_model_name: The name to save the trained model as.
    :param is_visualise_workflow: Whether to print the workflow graph.
    :param is_print_full_output: Whether to print the full output of the workflow.
    :return: The response from running the workflow.
    """
    predict_dataset_name = '_predict_data'
    is_upload_dataset = type(predict_dataset) is not str
    if is_upload_dataset:
        upload_dataset(
            client=client,
            project_name=project_name,
            dataset_name=predict_dataset_name,
            dataset=predict_dataset
        )
    else:
        predict_dataset_name = predict_dataset
    
    input_dataset = get_data(
        client=client,
        project_name=project_name,
        dataset_name=predict_dataset_name
    )
    if input_names is None:
        # input_names are the input_dataset columns
        input_names = input_dataset.columns.tolist()
    else:
        # Check that the input names exist in the dataset
        missing = set(input_names) - set(input_dataset.columns)
        if missing:
            raise ValueError(
                f"The following input_names are missing from dataset columns: {sorted(missing)}\n"
                f"Available columns: {input_dataset.columns.tolist()}"
            )
        
    # check the model's expected inputs
    expected_inputs = get_model_inputs(
        client=client,
        project_name=project_name,
        model_name=model_name,
    )
    missing = set(input_names) - set(expected_inputs)
    if missing:
        raise ValueError(
            f"The following input_names are missing from dataset columns: {sorted(missing)}\n"
            f"Available columns: {input_dataset.columns.tolist()}"
        )
    # 1. Create the graph
    graph = Graph()

    # 2. Create relevant nodes, handles, and add to graph:

    # 2.a. 
    
    # 2.a. Load dataset node
    load_data = Node(
        node_name="LoadDataset",
        label="Load Dataset",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name, 
                resource_name=predict_dataset_name, 
                resource_type='dataset')
                ),
        project_id=get_project_id(
            client=client,
            project_name=project_name
            ),
    )
    graph.add_node(load_data)  # add to graph
    dataset = load_data.make_handle("file")  # add handle
    if input_names is not None:
        # 2.b. Filter dataset node for inputs
        input_data = Node(
            node_name="FilterDataset",
            label="Input Dataset",
            columns=input_names,
            dataset=dataset,
        )
        graph.add_node(input_data)  # add to graph
        input_dataset = input_data.make_handle("dataset")  # add handle
    else:
        input_dataset = dataset
    
    # 2.a. Load model node
    load_model = Node(
        node_name="LoadModel",
        label="Load Model",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name, 
                resource_name=model_name, 
                resource_type='model')
                ),
        project_id=get_project_id(
            client=client,
            project_name=project_name
            ),
    )
    graph.add_node(load_model)  # add to graph
    # 2.d. Predict model node
    predict_model = Node(
        node_name="PredictModel",
        label="Predict Model",
        dataset=input_dataset,
        model=load_model.make_handle("file"),
    )
    graph.add_node(predict_model)  # add to graph
    
    # 2.e. Display node
    download_predict = Node(
        node_name="Download",
        label="Download Prediction",
        file=predict_model.make_handle("prediction"),
    )
    graph.add_node(download_predict)  # add to graph

    # 2.e. Display node
    download_uncertainty = Node(
        node_name="Download",
        label="Download Uncertainty",
        file=predict_model.make_handle("uncertainty"),
    )
    graph.add_node(download_uncertainty)  # add to graph

    if is_visualise_workflow:
        pprint(graph.nodes)

    workflow = Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        requested_output={
            "Predictions": download_predict.make_handle("file").model_dump(),
            "Uncertainty": download_uncertainty.make_handle("file").model_dump(),
            }
        )
    
    if save_workflow_name is not None:
        client.workflows.save(
            project_id=get_project_id(
                client=client,
                project_name=project_name
                ),
            workflow=workflow,
            workflow_name=save_workflow_name
        )

    response = client.run_node(workflow)
    if is_print_full_output:
        pprint(response.model_dump())

    # Download the predictions and save as a DataFrame
    predictions_response = get_presigned_url(response.outputs["outputs"]["Predictions"])
    predictions_df = pd.read_csv(StringIO(predictions_response.text))  # Save the predictions to a DataFrame

    # Download the uncertainty and save as a DataFrame
    uncertainty_response = get_presigned_url(response.outputs["outputs"]["Uncertainty"])
    uncertainty_df = pd.read_csv(StringIO(uncertainty_response.text))  # Save the uncertainty to a DataFrame

    # Clean up if the input dataset was uploaded within this function
    if is_upload_dataset:
        try:
            client.resources.delete_resource(
                project_id=get_project_id(
                    client=client,
                    project_name=project_name
                ),
                resource_id=get_resource_id(
                    client=client,
                    project_name=project_name,
                    resource_name=predict_dataset_name,
                    resource_type="dataset"
                ),
                resource_type="dataset",
            )
        except Exception as e:
            # Non-fatal: log or print if desired; don't mask main exceptions
            print(f"[WARN] Failed to delete temp dataset '{predict_dataset_name}': {e}")

    return predictions_df, uncertainty_df
