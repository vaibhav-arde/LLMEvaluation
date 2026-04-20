from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from src.llm import generate, OllamaEvalModel

def test_basic_qa():
    prompt = "What is the capital of India?"
    expected = "New Delhi"

    actual_output = generate(prompt)

    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=expected,
    )

    eval_model = OllamaEvalModel()
    metric = AnswerRelevancyMetric(threshold=0.7, model=eval_model)

    assert_test(test_case, [metric])
