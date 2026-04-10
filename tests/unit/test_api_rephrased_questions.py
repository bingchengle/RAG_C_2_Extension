"""Ensure comparative rephrase forwards the model to the underlying processor."""

from src.api_requests import APIProcessor


def test_get_rephrased_questions_forwards_model(monkeypatch):
    api = APIProcessor(provider="openai")
    captured: dict = {}

    def fake_send_message(model=None, **kwargs):
        captured["model"] = model
        return {"questions": [{"company_name": "Acme", "question": "What is Acme revenue?"}]}

    monkeypatch.setattr(api.processor, "send_message", fake_send_message)
    out = api.get_rephrased_questions("Compare A and B", ["Acme"], model="gpt-4o-mini")
    assert captured["model"] == "gpt-4o-mini"
    assert out["Acme"] == "What is Acme revenue?"
