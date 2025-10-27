from ue_helper.workflow_templates import train_and_save_model_workflow, predict_model_workflow, get_model_maximum_workflow, get_model_recommended_points_workflow

from uncertainty_engine import Client

from dotenv import load_dotenv
import pandas as pd

# automatically load .env in current directory
load_dotenv()

client = Client()
client.authenticate()

PROJECT_NAME = 'Personal'
MODEL_NAME = 'example'

inputs = {
    'x0': [0., 1.],
    'x1': [0., 1.],
}

outputs = {
    'y0': [0., 1.]
}

train_dataset = {**inputs, **outputs}
input_names = list(inputs.keys())
output_names = list(outputs.keys())

# train_and_save_model_workflow(
#     client=client,
#     project_name=PROJECT_NAME,
#     save_model_name=MODEL_NAME,
#     input_names=input_names,
#     output_names=output_names,
#     train_dataset=train_dataset,
# )

# prediction, uncertainty = predict_model_workflow(
#     client=client,
#     predict_dataset={
#         'x0': [0.5],
#         'x1': [0.5]
#     },
#     project_name=PROJECT_NAME,
#     model_name=MODEL_NAME,

# )
# print(prediction)
# print(uncertainty)

rec = get_model_recommended_points_workflow(
    client=client,
    project_name=PROJECT_NAME,
    model_name=MODEL_NAME,
    number_of_points=5,
    )

print(rec)

client = Client()
client.authenticate()

PROJECT_NAME = 'Personal'
MODEL_NAME = 'example'
pred, unc = get_model_maximum_workflow(
    client=client,
    project_name=PROJECT_NAME,
    model_name=MODEL_NAME,
    )
print(pred)
print(unc)