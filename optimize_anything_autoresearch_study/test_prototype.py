from optimized_anything_autoresearch import ResearchOptimizer, ToolAdapter


def test_optimizer_generates_sections_and_trace():
    optimizer = ResearchOptimizer(adapter=ToolAdapter(), max_iters=6, seed=7)
    state, trace = optimizer.run("自动化研究代理")

    assert len(trace) > 0
    assert len(state.subquestions) > 0
    assert len(state.draft_sections) > 0
    assert any("total" in step for step in trace)
