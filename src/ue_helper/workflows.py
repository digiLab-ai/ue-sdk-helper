from __future__ import annotations
import numpy as np
from pprint import pprint
from uncertainty_engine.nodes.base import Node
from uncertainty_engine.graph import Graph
from uncertainty_engine.nodes.workflow import Workflow
from .utils import get_presigned_url, get_project_id
from .active_learning import active_learning_step, active_learning_loop

__all__ = ["get_presigned_url", "get_project_id", "active_learning_step", "active_learning_loop"]
__version__ = "0.0.1"


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
    Simplified methods exist for local data.
    """
    # Create the graph
    graph = Graph()

    # Create relevant nodes:

    # Define the model config node
    model_config = Node(
        node_name="ModelConfig",
        label="Model Config",
    )
    graph.add_node(model_config)
    # Add a handle to the the config output
    output_config = model_config.make_handle("config")

    # Define the load dataset node
    load_data = Node(
        node_name="LoadDataset",
        label="Load Dataset",
        file_id=get_resource_id(client=client, project_name=project_name, resource_name=dataset_name),
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(load_data)
    dataset = load_data.make_handle("dataset")

    # Define the input filter dataset node
    input_data = Node(
        node_name="FilterDataset",
        label="Input Dataset",
        columns=input_names,
        dataset=dataset,
    )
    graph.add_node(input_data)
    input_dataset = load_data.make_handle("input_dataset")

    # Define the output filter dataset node
    output_data = Node(
        node_name="FilterDataset",
        label="Output Dataset",
        columns=output_names,
        dataset=dataset,
    )
    graph.add_node(output_data)
    output_dataset = load_data.make_handle("output_dataset")


    # Define the train model node
    train_model = Node(
        node_name="TrainModel",
        label="Train Model",
        config=output_config,
        inputs=input_dataset,
        outputs=output_dataset,
    )
    graph.add_node(train_model)
    # Add a handle to the the config output
    output_model = train_model.make_handle("model")
    save = Node(
        node_name="Save",
        label="Save",
        file=output_model,
        fileid=save_model_name,
    )
    graph.add_node(save)

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
        
