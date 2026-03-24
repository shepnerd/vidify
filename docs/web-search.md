# Web Search Integration

VidCopilot can enhance video analysis with web context using Google Custom Search API, with automatic fallback to Baidu for users in China.

## Features

- **Context enhancement** — automatically generates search queries from video content and retrieves relevant web information
- **Multi-region support** — auto-detects network environment, falls back to Baidu when Google is unreachable
- **Report integration** — web search results are incorporated into analysis reports
- **Optional** — disabled by default, does not affect existing functionality when unused

## Setup

### Google Custom Search API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable the Custom Search API
3. Create an API key and a Custom Search Engine
4. Set environment variables:

```bash
export GOOGLE_API_KEY="your_api_key_here"
export GOOGLE_SEARCH_ENGINE_ID="your_search_engine_id_here"
```

See [GOOGLE_SEARCH_SETUP.md](../GOOGLE_SEARCH_SETUP.md) for detailed instructions.

### Baidu Fallback

No setup required. When Google is unreachable, the system automatically falls back to Baidu web search. This provides a seamless experience for users in China or other regions with restricted internet access.

## Network Detection Logic

1. Attempt Google Custom Search (requires API key)
2. If Google is unreachable, switch to Baidu
3. If all external search fails, return local suggestions

## Usage

### CLI

```bash
# Brief analysis with web search
python agent/main.py youtube "https://www.youtube.com/watch?v=..." \
    --mode brief --include-web-search

# Report with web search and explicit credentials
python agent/main.py youtube "https://www.youtube.com/watch?v=..." \
    --mode report --include-web-search \
    --google-api-key "your_key" --google-search-engine-id "your_id"
```

### REST API

```bash
# Analysis with web search
curl -X POST http://localhost:9000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "source_type": "youtube",
    "uri": "https://www.youtube.com/watch?v=XXXX",
    "mode": "detailed",
    "include_web_search": true,
    "google_api_key": "your_key",
    "google_search_engine_id": "your_id"
  }'
```

## Testing

```bash
python tests/test_web_search.py             # Unit tests
python scripts/demo_multi_region_search.py  # Multi-region demo
```
