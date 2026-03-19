# Google Custom Search API Setup

To use the web search functionality in VidCopilot, you need to set up Google Custom Search API credentials.

## 1. Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one

## 2. Enable Custom Search API

1. In the Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Custom Search API" and enable it

## 3. Create API Key

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "API Key"
3. Copy the generated API key

## 4. Create Custom Search Engine

1. Go to [Google Custom Search](https://cse.google.com/)
2. Click "Add" to create a new search engine
3. Configure your search engine:
   - **Sites to search**: Leave blank for entire web, or specify domains
   - **Name**: Give it a descriptive name
   - **Language**: Choose your preferred language
4. Click "Create"
5. Go to "Control Panel" and copy the "Search engine ID" (cx parameter)

## 5. Set Environment Variables

Set the following environment variables:

```bash
export GOOGLE_API_KEY="your_api_key_here"
export GOOGLE_SEARCH_ENGINE_ID="your_search_engine_id_here"
```

Or pass them as command line arguments:

```bash
python agent/main.py --source-type youtube --uri <URL> --mode brief --include-web-search --google-api-key "your_api_key" --google-search-engine-id "your_search_engine_id"
```

## 6. Test the Setup

Run the test script to verify your credentials:

```bash
python test_web_search.py
```

## Usage Examples

### Brief analysis with web search:
```bash
python agent/main.py --source-type youtube --uri "https://www.youtube.com/watch?v=VIDEO_ID" --mode brief --include-web-search
```

### Detailed analysis with web search:
```bash
python agent/main.py --source-type youtube --uri "https://www.youtube.com/watch?v=VIDEO_ID" --mode detailed --include-web-search
```

### Generate comprehensive report:
```bash
python agent/main.py --source-type youtube --uri "https://www.youtube.com/watch?v=VIDEO_ID" --mode report --analysis-type detailed --include-web-search
```

## Multi-Region Support

VidCopilot automatically detects your network environment and provides the best search experience:

### For Global Users (Google Accessible)
- Uses Google Custom Search API for high-quality, comprehensive results
- Requires API key and search engine ID setup (see above)

### For Chinese Users (Google Blocked)
- **Automatic Fallback**: When Google is not accessible, automatically switches to Baidu search
- **No Configuration Required**: System detects network restrictions and adapts automatically
- **Local Fallback**: If all external searches fail, provides helpful local guidance

### Network Detection Logic
1. **Primary**: Try Google Custom Search (if credentials available and network allows)
2. **Secondary**: Fall back to Baidu search for Chinese users
3. **Tertiary**: Provide local search suggestions and usage guidance

### Testing Network Connectivity
```bash
python test_web_search.py
```

This will test your network connectivity and demonstrate the automatic fallback behavior.

## API Limits

- Google Custom Search API has daily limits (100 queries per day for free tier)
- Each search can return up to 10 results
- Monitor your usage in Google Cloud Console

## Troubleshooting

- **403 Forbidden**: Check your API key and search engine ID
- **Daily limit exceeded**: Upgrade to a paid plan or reduce search frequency
- **No results**: Verify your search engine configuration allows searching the entire web