import os

import pytest

from src.answer_generator_optimized import AnswerGeneratorOptimized


RUN_INTEGRATION = os.getenv("RUN_INTEGRATION_TESTS") == "1"
pytestmark = pytest.mark.integration

MOCK_CONTEXT = [
    {
        "text": "Company: Tesla, Inc\nYear: 2023\nRevenue: $96.8 billion\nNet Income: $15.0 billion\nEmployees: 140,000\nMarket Share: 18.5%\nStock Price: $248.48",
        "page": 1,
    },
    {"text": "R&D Expense: $3.2 billion\nVehicle Deliveries: 1.8 million", "page": 2},
]


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Set RUN_INTEGRATION_TESTS=1 to run API integration tests.")
def test_answer_generator_optimized_integration():
    generator = AnswerGeneratorOptimized()
    result = generator.generate_answer(query="特斯拉2023年营收是多少", context=MOCK_CONTEXT, schema="text")
    assert result.answer is not None
