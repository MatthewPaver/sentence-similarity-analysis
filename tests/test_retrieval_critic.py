from retrieval_critic import critique_retrieval


def test_retrieval_critic_passes_close_match():
    result = critique_retrieval("polar bear fur", ["A polar bear's fur is transparent", "Mars is red"])

    assert result.status == "pass"
    assert "polar bear" in result.top_match


def test_retrieval_critic_warns_on_weak_match():
    result = critique_retrieval("payroll automation", ["The moon is far away"])

    assert result.status == "review"
    assert "weak" in result.warning
