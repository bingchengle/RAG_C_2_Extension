"""Sanity checks for LLM reranker combined_score direction (vector distance → similarity)."""


def combined_score(llm_weight: float, relevance: float, distance: float) -> float:
    vector_weight = 1.0 - llm_weight
    vec_sim = max(0.0, min(1.0, 1.0 - float(distance)))
    return round(llm_weight * relevance + vector_weight * vec_sim, 4)


def test_same_relevance_higher_distance_ranks_lower():
    """With fixed LLM relevance, larger FAISS distance (less similar) must reduce combined_score."""
    lw = 0.7
    rel = 0.8
    good_vec = combined_score(lw, rel, 0.05)
    bad_vec = combined_score(lw, rel, 0.95)
    assert good_vec > bad_vec


def test_vector_similarity_clamped_to_unit_interval():
    assert combined_score(0.5, 1.0, -1.0) == round(0.5 * 1.0 + 0.5 * 1.0, 4)
    assert combined_score(0.5, 1.0, 5.0) == round(0.5 * 1.0 + 0.5 * 0.0, 4)
