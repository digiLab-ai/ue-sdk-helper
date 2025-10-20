# conftest.py
import os
import pytest
from dotenv import load_dotenv

load_dotenv()  # load .env at repo root (or wherever pytest is run)

@pytest.fixture(scope="session")
def ue_client():
    try:
        from uncertainty_engine import Client
    except Exception as e:
        pytest.skip(f"uncertainty_engine not installed: {e}")

    client = Client()
    try:
        client.authenticate()
    except Exception as e:
        pytest.skip(f"Unable to authenticate UE client (check .env): {e}")
    return client
