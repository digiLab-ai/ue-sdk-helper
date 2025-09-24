from __future__ import annotations
import numpy as np
import random
import os
import requests

def get_presigned_url(url):
    """
    Get the contents from the presigned url.
    """
    url = url.replace("https://", "http://")
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response

def get_project_id(client, project_name: str) -> str:
    """
    Get the project ID from the project name.
    """
    projects = client.projects.list_projects()
    for project in projects:
        if project.name == project_name:
            return project.id
    raise ValueError(f"Project with name {project_name} not found.")

def get_file_id(client, filename: str, project_name:str) -> str:
    """
    Get the file ID from the project name.
    """
    resources = client.resources.list_resources(project_id=get_project_id(client, project_name))
    for resource in resources:
        if resource.name == filename:
            return resource.id
    raise ValueError(f"Project with name {filename} not found.")

