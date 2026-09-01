import pandas as pd
import pytest


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "sex": ["1", "1", "2", "1", None, "2", "2", "1", "2", "2"],
            "age_group": ["2", "1", "2", "3", "1", "2", "2", "2", "3", None],
            "region": [None, "2", "1", "2", "2", "3", "3", "2", "2", "1"],
            "education": ["3", "2", "3", "1", "3", "3", "3", "1", "3", "2"],
            "income": [300, 100, 450, 200, 650, 750, None, 850, 400, 350],
            "tax": [100, 10, 200, 90, 340, 370, 30, None, 150, 150],
            "weight": [1.5, 3.2, 1.7, 2.2, 6.1, 4.2, 1.9, 4.8, None, 8.2],
        }
    )


@pytest.fixture
def labels():
    return {
        "sex": {"1": "Males", "2": "Females"},
        "age_group": {"1": "0-19", "2": "20-66", "3": "67+"},
        "region": {"1": "West", "2": "East", "3": "Central"},
        "education": {
            "3": "Higher education",
            "2": "Secondary school",
            "1": "Elementary school",
        },
    }
