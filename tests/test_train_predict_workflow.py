# test_train_predict_workflow.py
import math
import pandas as pd
import pytest

from ue_helper.workflow_templates import (
    train_and_save_model_workflow,
    predict_model_workflow,
)

PROJECT_NAME = "Personal"
MODEL_NAME = "example_pytest_model"


@pytest.mark.integration
def test_train_and_predict_roundtrip(ue_client):
    # --- Prepare minimal training data ---
    inputs = {
        "x0": [0.0, 1.0],
        "x1": [0.0, 1.0],
    }
    outputs = {
        "y0": [0.0, 1.0],
    }
    train_dataset = {**inputs, **outputs}

    # --- Train and save model ---
    resp = train_and_save_model_workflow(
        client=ue_client,
        project_name=PROJECT_NAME,
        save_model_name=MODEL_NAME,
        input_names=list(inputs.keys()),
        output_names=list(outputs.keys()),
        train_dataset=train_dataset,
        is_visualise_workflow=False,
        is_print_full_output=False,
        save_workflow_name=None,
    )
    # Sanity: response object present and has an id-ish field (don’t over-specify)
    assert resp is not None

    # --- Predict with the trained model ---
    # --- Predict on a tiny input set ---
    predict_inputs = {"x0": [0.5], "x1": [0.5]}
    prediction_df, uncertainty_df = predict_model_workflow(
        client=ue_client,
        predict_dataset=predict_inputs,
        project_name=PROJECT_NAME,
        model_name=MODEL_NAME,
        input_names=list(predict_inputs.keys()),
        is_visualise_workflow=False,
        is_print_full_output=False,
        save_workflow_name=None,
    )
    print(prediction_df)
    print(uncertainty_df)
