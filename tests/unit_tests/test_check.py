import pytest

@pytest.fixture
def example_fixture():
    return 0

def test_with_fixture(example_fixture):
    assert example_fixture == 0