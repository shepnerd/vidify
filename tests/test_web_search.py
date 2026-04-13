from agent.extensions.skills.web_search import deep_search_enhance, web_search


def test_web_search_prefers_google_when_available(monkeypatch):
    expected = [{"title": "Result", "url": "https://example.com", "snippet": "x", "source": "google"}]

    monkeypatch.setattr("agent.extensions.skills.web_search.is_google_accessible", lambda: True)
    monkeypatch.setattr("agent.extensions.skills.web_search.google_search", lambda *args, **kwargs: expected)

    results = web_search("video analysis", api_key="key", search_engine_id="cx")

    assert results == expected


def test_web_search_falls_back_to_local_when_google_and_baidu_fail(monkeypatch):
    fallback = [{"title": "Fallback", "url": "README.md", "snippet": "offline", "source": "local_fallback"}]

    monkeypatch.setattr("agent.extensions.skills.web_search.is_google_accessible", lambda: False)
    monkeypatch.setattr("agent.extensions.skills.web_search.baidu_search", lambda *args, **kwargs: [])
    monkeypatch.setattr("agent.extensions.skills.web_search.local_search_fallback", lambda *args, **kwargs: fallback)

    results = web_search("video analysis")

    assert results == fallback


def test_deep_search_enhance_combines_query_and_context(monkeypatch):
    captured = {}

    def _fake_search(query, num_results, api_key, search_engine_id):
        captured["query"] = query
        captured["num_results"] = num_results
        return [{"title": "Result", "url": "https://example.com", "snippet": "x", "source": "google"}]

    monkeypatch.setattr("agent.extensions.skills.web_search.web_search", _fake_search)

    result = deep_search_enhance("machine learning", "video content analysis", num_results=2)

    assert captured == {
        "query": "machine learning video content analysis",
        "num_results": 2,
    }
    assert result["search_query"] == "machine learning video content analysis"
    assert len(result["search_results"]) == 1
