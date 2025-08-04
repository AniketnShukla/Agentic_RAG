import pytest
from src.agents.generator import Generator

# A fixture to initialize the generator with a mock model can be added here later.

def test_generator_initialization():
    """
    Tests that the Generator agent can be initialized.
    """
    mock_model = None
    generator = Generator(model=mock_model)
    assert generator is not None

def test_generate_with_dummy_data():
    """
    Tests the generate method with a simple query and context.
    """
    mock_model = None
    generator = Generator(model=mock_model)
    query = "What is the capital of France?"
    documents = ["Paris is the capital of France.", "The Eiffel Tower is in Paris."]
    answer = generator.generate(query, documents)

    assert isinstance(answer, str)
    assert "Paris" in answer
    assert "France" in answer
