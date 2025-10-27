from __future__ import annotations

import numpy as np
from pprint import pprint
from typing import Any, Dict, Optional, Union, Iterable, Tuple, Literal
import pandas as pd
from io import StringIO

from uncertainty_engine.nodes.base import Node
from uncertainty_engine.graph import Graph
from uncertainty_engine.nodes.workflow import Workflow
from uncertainty_engine import Client
from ue_helper.utils import (
    get_project_id,
    get_resource_id,
    wrap_resource_id,
    get_presigned_url,
    get_model_inputs,
    get_data,
    upload_dataset,
)

def train_and_save_model_workflow(
    client: Client,
    project_name: str,
    save_model_name: str,
    input_names: list,
    output_names: list,
    train_dataset: Optional[Union[str, pd.DataFrame, Dict[str, Iterable[Any]]]],
    is_visualise_workflow: bool = False,
    is_print_full_output: bool = False,
    save_workflow_name: Union[str, None] = None,
) -> dict:
    """
    Build and execute a workflow that *trains* a model on a provided dataset and
    saves the fitted model to the project.

    The dataset can be passed directly (as a pandas DataFrame or a column-oriented
    dict of iterables) or referenced by the name of an existing uploaded dataset.
    If a non-string object is provided, it is uploaded under a temporary name
    (`'_train_data'` by default) for the duration of this call.

    Parameters
    ----------
    client: Client
        Uncertainty Engine client instance (already authenticated).
    project_name : str
        Name of the target project that contains/receives all resources.
    save_model_name : str
        Resource name under which to persist the trained model.
    input_names : list
        Column names to use as *inputs* (X) for training.
    output_names : list
        Column names to use as *outputs* (y/targets) for training.
    train_dataset : str or pandas.DataFrame or dict[str, Iterable]
        - If `str`: name of an existing dataset resource in the project.
        - If `DataFrame` or dict-like: data to upload as a temporary dataset.
    is_visualise_workflow : bool, default False
        If True, pretty-print the constructed graph nodes for debugging.
    is_print_full_output : bool, default False
        If True, pretty-print the full response payload from workflow execution.
    save_workflow_name : str or None, default None
        If provided, the built workflow is saved to the project under this name.

    Returns
    -------
    dict
        The SDK response (as a dict-like) from running the training workflow.

    Raises
    ------
    ValueError
        If any of `input_names` or `output_names` are not present in the dataset.
    RuntimeError
        Propagated from the underlying client if workflow execution fails.

    Notes
    -----
    - This function assumes required nodes/operators ("LoadDataset", "FilterDataset",
      "TrainModel", "Save", "ModelConfig") exist in your backend.
    - All resources are assumed to be in the same project given by `project_name`.

    Examples
    --------
    >>> train_and_save_model_workflow(
    ...     client=ue_client,
    ...     project_name="Personal",
    ...     save_model_name="gp_emulator",
    ...     input_names=["x0", "x1"],
    ...     output_names=["y0"],
    ...     train_dataset=df  # a pandas DataFrame
    ... )
    """
    # Determine whether we need to upload a local dataset or reference an existing one.
    train_dataset_name = "_train_data"
    is_upload_dataset = type(train_dataset) is not str  # explicit, avoids isinstance confusion
    if is_upload_dataset:
        upload_dataset(
            client=client,
            project_name=project_name,
            dataset_name=train_dataset_name,
            dataset=train_dataset,
        )
    else:
        # Use the provided name as-is (dataset already exists in the project).
        train_dataset_name = train_dataset  # type: ignore[assignment]

    # Resolve the dataset into a local DataFrame for schema validation.
    train_dataset_df = get_data(
        client=client, project_name=project_name, dataset_name=train_dataset_name
    )

    # Validate requested columns exist.
    missing_inputs = set(input_names) - set(train_dataset_df.columns)
    if missing_inputs:
        raise ValueError(
            "The following `input_names` are missing from dataset columns: "
            f"{sorted(missing_inputs)}\nAvailable columns: {train_dataset_df.columns.tolist()}"
        )

    missing_outputs = set(output_names) - set(train_dataset_df.columns)
    if missing_outputs:
        raise ValueError(
            "The following `output_names` are missing from dataset columns: "
            f"{sorted(missing_outputs)}\nAvailable columns: {train_dataset_df.columns.tolist()}"
        )

    # 1) Construct the workflow graph.
    graph = Graph()

    # 2a) Model configuration node (hyperparameters, architecture, etc.).
    model_config = Node(
        node_name="ModelConfig",
        label="Model Config",
    )
    graph.add_node(model_config)

    # 2b) Load dataset node (points at the dataset resource in the project).
    load_data = Node(
        node_name="LoadDataset",
        label="Load Dataset",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name,
                resource_name=train_dataset_name,
                resource_type="dataset",
            )
        ),
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(load_data)

    # 2c) Filter to training inputs (X).
    input_data = Node(
        node_name="FilterDataset",
        label="Input Dataset",
        columns=input_names,
        dataset=load_data.make_handle("file"),
    )
    graph.add_node(input_data)

    # 2d) Filter to training outputs (y).
    output_data = Node(
        node_name="FilterDataset",
        label="Output Dataset",
        columns=output_names,
        dataset=load_data.make_handle("file"),
    )
    graph.add_node(output_data)

    # 2e) Train the model.
    train_model = Node(
        node_name="TrainModel",
        label="Train Model",
        config=model_config.make_handle("config"),
        inputs=input_data.make_handle("dataset"),
        outputs=output_data.make_handle("dataset"),
    )
    graph.add_node(train_model)

    # 2f) Persist the trained model as a project resource.
    save = Node(
        node_name="Save",
        label="Save",
        data=train_model.make_handle("model"),
        file_id=save_model_name,
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(save)

    if is_visualise_workflow:
        pprint(graph.nodes)

    # Finalize workflow payload.
    workflow = Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        requested_output={},  # training typically yields model artifact persisted via Save node
    )

    # Optionally persist the workflow definition itself (for reuse/auditing).
    if save_workflow_name is not None:
        client.workflows.save(
            project_id=get_project_id(client=client, project_name=project_name),
            workflow=workflow,
            workflow_name=save_workflow_name,
        )

    # Execute training.
    response = client.run_node(workflow)

    if is_print_full_output:
        pprint(response.model_dump())

    return response  # SDK object is dict-like; keep as-is for caller flexibility


def predict_model_workflow(
    client: Client,
    predict_dataset: Optional[Union[str, pd.DataFrame, Dict[str, Iterable[Any]]]],
    project_name: str,
    model_name: str,
    input_names: Union[list, None] = None,
    is_visualise_workflow: bool = False,
    is_print_full_output: bool = False,
    save_workflow_name: Union[str, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and execute a workflow that *loads a saved model* and produces predictions
    (and associated uncertainties) on a provided dataset.

    The dataset can be passed directly (as a pandas DataFrame or a column-oriented
    dict of iterables) or referenced by name if already uploaded. If a non-string
    object is provided, it is uploaded under a temporary name (`'_predict_data'`)
    for the duration of this call and then deleted.

    Parameters
    ----------
    client : Client
        Uncertainty Engine client instance (already authenticated).
    predict_dataset : str or pandas.DataFrame or dict[str, Iterable]
        - If `str`: name of an existing dataset resource in the project.
        - If `DataFrame` or dict-like: data to upload as a temporary dataset.
    project_name : str
        Name of the project where the dataset/model resources live.
    model_name : str
        Name of the saved model resource to load for inference.
    input_names : list or None, default None
        Subset of columns to feed to the model. If `None`, all columns in the
        provided dataset are used (in their existing order). The chosen set is
        validated against the model's expected inputs.
    is_visualise_workflow : bool, default False
        If True, pretty-print the constructed graph nodes for debugging.
    is_print_full_output : bool, default False
        If True, pretty-print the full response payload from workflow execution.
    save_workflow_name : str or None, default None
        If provided, the built workflow is saved to the project under this name.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        A pair `(predictions_df, uncertainty_df)` loaded from the workflow's
        downloadable artifacts.

    Raises
    ------
    ValueError
        - If requested `input_names` are not found in the dataset.
        - If `input_names` do not match a subset of the model's expected inputs.
    RuntimeError
        Propagated from the underlying client if workflow execution fails.

    Notes
    -----
    - This function assumes the nodes/operators "LoadDataset", "FilterDataset",
      "LoadModel", "PredictModel", and "Download" are available.
    - Temporary datasets uploaded by this function are deleted at the end,
      with failures logged as warnings rather than raising.

    Examples
    --------
    >>> preds, uncert = predict_model_workflow(
    ...     client=ue_client,
    ...     predict_dataset="validate",
    ...     project_name="Personal",
    ...     model_name="gp_emulator",
    ...     input_names=["x0", "x1"]
    ... )
    """
    # Resolve dataset: upload (temporary) or reference by name.
    predict_dataset_name = "_predict_data"
    is_upload_dataset = type(predict_dataset) is not str
    if is_upload_dataset:
        upload_dataset(
            client=client,
            project_name=project_name,
            dataset_name=predict_dataset_name,
            dataset=predict_dataset,
        )
    else:
        predict_dataset_name = predict_dataset  # type: ignore[assignment]

    # Pull as DataFrame for schema checks (and defaulting input_names).
    input_dataset_df = get_data(
        client=client, project_name=project_name, dataset_name=predict_dataset_name
    )

    # If caller didn't specify, default to "use all columns as model inputs".
    if input_names is None:
        input_names = input_dataset_df.columns.tolist()
    else:
        # Validate requested columns exist in the dataset.
        missing_cols = set(input_names) - set(input_dataset_df.columns)
        if missing_cols:
            raise ValueError(
                "The following `input_names` are missing from dataset columns: "
                f"{sorted(missing_cols)}\nAvailable columns: {input_dataset_df.columns.tolist()}"
            )

    # Validate against the model schema: the model must accept at least these inputs.
    expected_inputs = get_model_inputs(
        client=client, project_name=project_name, model_name=model_name
    )
    missing_for_model = set(input_names) - set(expected_inputs)
    if missing_for_model:
        raise ValueError(
            "The provided `input_names` are not accepted by the model. "
            f"Missing from model's expected inputs: {sorted(missing_for_model)}\n"
            f"Model expects: {sorted(expected_inputs)}"
        )

    # 1) Build graph.
    graph = Graph()

    # 2a) Load dataset resource.
    load_data = Node(
        node_name="LoadDataset",
        label="Load Dataset",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name,
                resource_name=predict_dataset_name,
                resource_type="dataset",
            )
        ),
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(load_data)
    dataset_handle = load_data.make_handle("file")

    # 2b) Optional filtering to the specified inputs (keeps column order given).
    if input_names is not None:
        input_data = Node(
            node_name="FilterDataset",
            label="Input Dataset",
            columns=input_names,
            dataset=dataset_handle,
        )
        graph.add_node(input_data)
        input_dataset_handle = input_data.make_handle("dataset")
    else:
        input_dataset_handle = dataset_handle  # pragma: no cover

    # 2c) Load the trained model.
    load_model = Node(
        node_name="LoadModel",
        label="Load Model",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name,
                resource_name=model_name,
                resource_type="model",
            )
        ),
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(load_model)

    # 2d) Predict with uncertainty.
    predict_model = Node(
        node_name="PredictModel",
        label="Predict Model",
        dataset=input_dataset_handle,
        model=load_model.make_handle("file"),
    )
    graph.add_node(predict_model)

    # 2e) Download artifacts (predictions + uncertainty as CSVs).
    download_predict = Node(
        node_name="Download",
        label="Download Prediction",
        file=predict_model.make_handle("prediction"),
    )
    graph.add_node(download_predict)

    download_uncertainty = Node(
        node_name="Download",
        label="Download Uncertainty",
        file=predict_model.make_handle("uncertainty"),
    )
    graph.add_node(download_uncertainty)

    if is_visualise_workflow:
        pprint(graph.nodes)

    # Finalize workflow payload (explicitly request both artifacts).
    workflow = Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        requested_output={
            "Predictions": download_predict.make_handle("file").model_dump(),
            "Uncertainty": download_uncertainty.make_handle("file").model_dump(),
        },
    )

    # Optionally persist the workflow definition.
    if save_workflow_name is not None:
        client.workflows.save(
            project_id=get_project_id(client=client, project_name=project_name),
            workflow=workflow,
            workflow_name=save_workflow_name,
        )

    # Execute inference.
    response = client.run_node(workflow)
    if is_print_full_output:
        pprint(response.model_dump())

    # Fetch artifacts via presigned URLs and load to DataFrames.
    predictions_response = get_presigned_url(response.outputs["outputs"]["Predictions"])
    predictions_df = pd.read_csv(StringIO(predictions_response.text))

    uncertainty_response = get_presigned_url(response.outputs["outputs"]["Uncertainty"])
    uncertainty_df = pd.read_csv(StringIO(uncertainty_response.text))

    # Clean up any temporary dataset uploaded by this function.
    if is_upload_dataset:
        try:
            client.resources.delete_resource(
                project_id=get_project_id(client=client, project_name=project_name),
                resource_id=get_resource_id(
                    client=client,
                    project_name=project_name,
                    resource_name=predict_dataset_name,
                    resource_type="dataset",
                ),
                resource_type="dataset",
            )
        except Exception as e:
            # Non-fatal: log warning but don't disrupt the main return path.
            print(f"[WARN] Failed to delete temp dataset '{predict_dataset_name}': {e}")

    return predictions_df, uncertainty_df


def get_model_recommended_points_workflow(
    client: Client,
    project_name: str,
    model_name: str,
    number_of_points: int,
    acquisition_function: Literal[
        "PosteriorStandardDeviation",
        "MonteCarloNegativeIntegratedPosteriorVariance",
    ] = "MonteCarloNegativeIntegratedPosteriorVariance",
    is_visualise_workflow: bool = False,
    is_print_full_output: bool = False,
    save_workflow_name: Union[str, None] = None,
) -> pd.DataFrame:
    """
    Build and execute a workflow that *loads a saved model* and asks it to
    recommend a set of new evaluation points using a chosen acquisition function.

    This creates a minimal graph with:
      - LoadModel  →  Recommend  →  Download

    Parameters
    ----------
    client : Client
        Authenticated Uncertainty Engine client.
    project_name : str
        Name of the project that contains the saved model.
    model_name : str
        Name of the saved model resource to load for recommendation.
    number_of_points : int
        How many candidate points the acquisition should return.
    acquisition_function : {"PosteriorStandardDeviation",
                            "MonteCarloNegativeIntegratedPosteriorVariance"}, optional
        Which acquisition strategy to use. Defaults to
        "MonteCarloNegativeIntegratedPosteriorVariance" (MC-NIPV), which seeks
        points that most reduce the integrated posterior variance.
        Brief notes:
          - ExpectedImprovement / LogExpectedImprovement: classic improvement-seeking
            exploitation/exploration balance (log variant for numerically wide ranges).
          - PosteriorMean: greedily picks high mean predictions (pure exploitation).
          - PosteriorStandardDeviation: targets high uncertainty (pure exploration).
          - MonteCarloExpectedImprovement / MonteCarloLogExpectedImprovement:
            MC estimators of EI / log-EI for non-Gaussian or complex posteriors.
          - MonteCarloNegativeIntegratedPosteriorVariance:
            MC estimator that minimizes global uncertainty (space-filling / UQ-oriented).
    is_visualise_workflow : bool, default False
        If True, pretty-print the constructed graph nodes for debugging.
    is_print_full_output : bool, default False
        If True, pretty-print the full response payload from workflow execution.
    save_workflow_name : str or None, default None
        If provided, saves the built workflow under this name in the project.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of recommended points downloaded from the "Recommend" node
        (typically one row per suggested point; columns depend on the model's
        input schema and engine output).

    Raises
    ------
    RuntimeError
        Propagated from the underlying client if workflow execution fails.
    ValueError
        May be raised upstream if the provided acquisition name is not supported
        by the engine (this function passes it through as-is).

    Notes
    -----
    - Requires the engine operators: "LoadModel", "Recommend", and "Download".
    - The acquisition function string is forwarded directly to the engine.
      Use one of the listed values above to avoid runtime errors.

    Examples
    --------
    >>> df = get_model_recommended_points_workflow(
    ...     client=ue_client,
    ...     project_name="Personal",
    ...     model_name="gp_emulator",
    ...     number_of_points=5,
    ...     acquisition_function="MonteCarloExpectedImprovement",
    ... )
    >>> df.head()
    """

    # 1) Build graph.
    graph = Graph()

    # 2a) Load the trained model.
    load_model = Node(
        node_name="LoadModel",
        label="Load Model",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name,
                resource_name=model_name,
                resource_type="model",
            )
        ),
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(load_model)

    # 2d) Predict with uncertainty.
    recommend_model = Node(
        node_name="Recommend",
        label="Recommend",
        model=load_model.make_handle("file"),
        acquisition_function=acquisition_function,
        number_of_points=number_of_points,
    )
    graph.add_node(recommend_model)

    # 2e) Download artifacts (predictions + uncertainty as CSVs).
    download_recommended_points = Node(
        node_name="Download",
        label="Recommended points",
        file=recommend_model.make_handle("recommended_points"),
    )
    graph.add_node(download_recommended_points)

    if is_visualise_workflow:
        pprint(graph.nodes)

    # Finalize workflow payload (explicitly request both artifacts).
    workflow = Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        requested_output={
            "Recommended Points": download_recommended_points.make_handle("file").model_dump(),
        },
    )

    # Optionally persist the workflow definition.
    if save_workflow_name is not None:
        client.workflows.save(
            project_id=get_project_id(client=client, project_name=project_name),
            workflow=workflow,
            workflow_name=save_workflow_name,
        )

    # Execute inference.
    response = client.run_node(workflow)
    if is_print_full_output:
        pprint(response.model_dump())

    # Fetch artifacts via presigned URLs and load to DataFrames.
    recommended_points_response = get_presigned_url(response.outputs["outputs"]["Recommended Points"])
    recommended_points_response = pd.read_csv(StringIO(recommended_points_response.text))

    return recommended_points_response

def get_model_maximum_workflow(
    client: Client,
    project_name: str,
    model_name: str,
    number_of_points: int = 1,
    acquisition_function: Literal[
        "ExpectedImprovement",
        "LogExpectedImprovement",
        "PosteriorMean",
        "MonteCarloExpectedImprovement",
        "MonteCarloLogExpectedImprovement",
    ] = "PosteriorMean",
    is_visualise_workflow: bool = False,
    is_print_full_output: bool = False,
    save_workflow_name: Union[str, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and execute a workflow that loads a saved model, proposes up to
    `number_of_points` locations that *maximize the objective* according to the
    chosen acquisition function, and then evaluates the model at those points to
    return predictions and uncertainties.

    The constructed workflow is:
        1) LoadModel
        2) Recommend(acquisition_function, number_of_points)
        3) PredictModel(dataset = recommended_points, model = loaded model)
        4) Download(prediction), Download(uncertainty)

    Parameters
    ----------
    client : Client
        Uncertainty Engine client instance (already authenticated).
    project_name : str
        Name of the project containing the saved model.
    model_name : str
        Name of the saved model resource to load for recommendation and prediction.
    number_of_points : int, default 1
        How many candidate maximizers to propose.
    acquisition_function : {"ExpectedImprovement", "LogExpectedImprovement",
                            "PosteriorMean",
                            "MonteCarloExpectedImprovement", "MonteCarloLogExpectedImprovement"},
        default "PosteriorMean"
        Acquisition used to propose maximizers:
        - **ExpectedImprovement (EI):** Classic improvement over current best (greedy-exploit with exploration).
        - **LogExpectedImprovement:** Log-space EI, numerically stabler for tiny improvements.
        - **PosteriorMean:** Greedy exploitation of the model mean (argmax of mean).
        - **MonteCarloExpectedImprovement (MC-EI):** EI estimated via posterior MC samples (better for batches/multi-modality).
        - **MonteCarloLogExpectedImprovement (MC-log-EI):** Log-space MC EI for numerical stability.

    is_visualise_workflow : bool, default False
        If True, pretty-print the constructed graph nodes for debugging.
    is_print_full_output : bool, default False
        If True, pretty-print the full response payload from workflow execution.
    save_workflow_name : str or None, default None
        If provided, saves the workflow definition under this name in the project.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Tuple of `(predictions_df, uncertainty_df)` evaluated at the recommended points.
        Columns reflect the model outputs and their associated uncertainties.

    Raises
    ------
    ValueError
        If `acquisition_function` is not one of the supported options.
    RuntimeError
        Propagated from the underlying client if workflow execution fails.

    Notes
    -----
    - This routine targets **maximization**. For minimization, maximize the negative
      of the objective (or use an equivalent acquisition configured for minimization).
    - Assumes availability of "LoadModel", "Recommend", "PredictModel", and "Download" nodes.

    Examples
    --------
    >>> preds_df, uncert_df = get_model_maximum_workflow(
    ...     client=ue_client,
    ...     project_name="Personal",
    ...     model_name="gp_emulator",
    ...     number_of_points=5,
    ...     acquisition_function="ExpectedImprovement",
    ... )
    >>> preds_df.head(), uncert_df.head()
    """
    # 1) Build graph.
    graph = Graph()

    # 2a) Load the trained model.
    load_model = Node(
        node_name="LoadModel",
        label="Load Model",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name,
                resource_name=model_name,
                resource_type="model",
            )
        ),
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(load_model)

    # 2d) Predict with uncertainty.
    recommend_model = Node(
        node_name="Recommend",
        label="Recommend",
        model=load_model.make_handle("file"),
        acquisition_function=acquisition_function,
        number_of_points=number_of_points,
    )
    graph.add_node(recommend_model)

    # 2d) Predict with uncertainty.
    predict_model = Node(
        node_name="PredictModel",
        label="Predict Model",
        dataset=recommend_model.make_handle("recommended_points"),
        model=load_model.make_handle("file"),
    )
    graph.add_node(predict_model)

    # 2e) Download artifacts (predictions + uncertainty as CSVs).
    download_predict = Node(
        node_name="Download",
        label="Download Prediction",
        file=predict_model.make_handle("prediction"),
    )
    graph.add_node(download_predict)

    download_uncertainty = Node(
        node_name="Download",
        label="Download Uncertainty",
        file=predict_model.make_handle("uncertainty"),
    )
    graph.add_node(download_uncertainty)

    if is_visualise_workflow:
        pprint(graph.nodes)

    # Finalize workflow payload (explicitly request both artifacts).
    workflow = Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        requested_output={
            "Predictions": download_predict.make_handle("file").model_dump(),
            "Uncertainty": download_uncertainty.make_handle("file").model_dump(),
        },
    )

    # Optionally persist the workflow definition.
    if save_workflow_name is not None:
        client.workflows.save(
            project_id=get_project_id(client=client, project_name=project_name),
            workflow=workflow,
            workflow_name=save_workflow_name,
        )

    # Execute inference.
    response = client.run_node(workflow)
    if is_print_full_output:
        pprint(response.model_dump())

    # Fetch artifacts via presigned URLs and load to DataFrames.
    predictions_response = get_presigned_url(response.outputs["outputs"]["Predictions"])
    predictions_df = pd.read_csv(StringIO(predictions_response.text))

    uncertainty_response = get_presigned_url(response.outputs["outputs"]["Uncertainty"])
    uncertainty_df = pd.read_csv(StringIO(uncertainty_response.text))

    return predictions_df, uncertainty_df
