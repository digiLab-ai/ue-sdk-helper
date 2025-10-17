from __future__ import annotations
import numpy as np
from pprint import pprint
from uncertainty_engine.nodes.base import Node
from uncertainty_engine.graph import Graph
from uncertainty_engine.nodes.workflow import Workflow

from ue_helper.utils import (
    get_project_id,
    get_resource_id,
    wrap_resource_id,
)

def train_and_save_model_workflow(client, 
                   project_name: str,
                   dataset_name: str, 
                   input_names: list,
                   output_names: list,
                   save_model_name: str,
                   is_visualise_workflow: bool = False,
                   is_print_full_output: bool = False) -> dict:
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
    # 1. Create the graph
    graph = Graph()

    # 2. Create relevant nodes, handles, and add to graph:

    # 2.a. Model config node
    model_config = Node(
        node_name="ModelConfig",
        label="Model Config",
    )
    graph.add_node(model_config)  # add to graph
    output_config = model_config.make_handle("config")  # add handle
    
    # 2.b. Load dataset node
    load_data = Node(
        node_name="LoadDataset",
        label="Load Dataset",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name, 
                resource_name=dataset_name, 
                resource_type='dataset')
                ),
        project_id=get_project_id(
            client=client,
            project_name=project_name
            ),
    )
    graph.add_node(load_data)  # add to graph
    dataset = load_data.make_handle("file")  # add handle
    
    # 2.b. Filter dataset node for inputs
    input_data = Node(
        node_name="FilterDataset",
        label="Input Dataset",
        columns=input_names,
        dataset=dataset,
    )
    graph.add_node(input_data)  # add to graph
    input_dataset = input_data.make_handle("dataset")  # add handle
    
    # 2.c. Filter dataset node for outputs
    output_data = Node(
        node_name="FilterDataset",
        label="Output Dataset",
        columns=output_names,
        dataset=dataset,
    )
    graph.add_node(output_data)  # add to graph
    output_dataset = output_data.make_handle("dataset")  # add handle
    
    # 2.d. Train model node
    train_model = Node(
        node_name="TrainModel",
        label="Train Model",
        config=output_config,
        inputs=input_dataset,
        outputs=output_dataset,
    )
    graph.add_node(train_model)  # add to graph
    output_model = train_model.make_handle("model")  # add handle
    
    # 2.e. Save model node
    save = Node(
        node_name="Save",
        label="Save",
        data=output_model,
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

    response = client.run_node(workflow)
    if is_print_full_output:
        pprint(response.model_dump())
        