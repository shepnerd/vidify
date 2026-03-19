# agent/skills/web_search.py
import requests
import os
import socket
from typing import List, Dict, Optional
from urllib.parse import quote

def check_network_connectivity(host: str, port: int = 80, timeout: float = 5) -> bool:
    """
    Check if a host is reachable.
    
    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Connection timeout in seconds
        
    Returns:
        True if host is reachable, False otherwise
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def is_google_accessible() -> bool:
    """
    Check if Google services are accessible from current network.
    
    Returns:
        True if Google is accessible, False otherwise
    """
    # Try multiple Google domains
    google_hosts = [
        ('www.google.com', 80),
        ('www.googleapis.com', 443),
        ('www.google.com.hk', 80)
    ]
    
    for host, port in google_hosts:
        if check_network_connectivity(host, port, 3):
            return True
    return False

def baidu_search(query: str, num_results: int = 5) -> List[Dict]:
    """
    Perform a web search using Baidu search (for Chinese users).
    
    Args:
        query: The search query string
        num_results: Number of results to return
        
    Returns:
        List of search results with title, url, snippet
    """
    try:
        # Baidu search URL
        search_url = f"https://www.baidu.com/s?wd={quote(query)}&rn={num_results}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse Baidu search results (simplified parsing)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        # Find search result containers
        result_containers = soup.find_all('div', class_='result')
        
        for container in result_containers[:num_results]:
            title_elem = container.find('h3')
            link_elem = container.find('a')
            desc_elem = container.find('div', class_='c-abstract')
            
            if title_elem and link_elem:
                title = title_elem.get_text().strip()
                url = link_elem.get('href', '')
                snippet = desc_elem.get_text().strip() if desc_elem else ''
                
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet,
                    'source': 'baidu'
                })
        
        return results
    
    except Exception as e:
        print(f"Baidu search failed: {e}")
        return []

def local_search_fallback(query: str, num_results: int = 5) -> List[Dict]:
    """
    Provide local search fallback when external search is not available.
    
    Args:
        query: The search query string
        num_results: Number of results to return
        
    Returns:
        List of mock search results with helpful information
    """
    # Provide helpful fallback information
    fallback_results = [
        {
            'title': f'关于 "{query}" 的本地搜索提示',
            'url': 'https://www.baidu.com',
            'snippet': f'由于网络限制，无法访问外部搜索服务。请尝试使用百度搜索 "{query}" 获取相关信息。',
            'source': 'local_fallback'
        },
        {
            'title': '网络连接问题解决方案',
            'url': 'https://www.google.com',
            'snippet': '如果您在中国大陆，可能需要使用VPN或代理服务来访问Google服务。或者使用百度等本地搜索引擎。',
            'source': 'local_fallback'
        },
        {
            'title': 'VidCopilot 功能说明',
            'url': 'README.md',
            'snippet': 'VidCopilot 支持多种分析模式，包括简要分析、详细分析、高光剪辑等。即使没有网络搜索功能，基本的视频分析功能仍然可用。',
            'source': 'local_fallback'
        }
    ]
    
    return fallback_results[:num_results]

def web_search(query: str, num_results: int = 5, api_key: str = None, search_engine_id: str = None) -> List[Dict]:
    """
    Perform a web search with automatic fallback for different regions.
    
    Args:
        query: The search query string
        num_results: Number of results to return (max 10 per request)
        api_key: Google Custom Search API key
        search_engine_id: Google Custom Search Engine ID
        
    Returns:
        List of search results with title, url, snippet
    """
    # First, try Google if credentials are available and network allows
    if api_key or os.getenv('GOOGLE_API_KEY'):
        if not api_key:
            api_key = os.getenv('GOOGLE_API_KEY')
        if not search_engine_id:
            search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        if api_key and search_engine_id and is_google_accessible():
            try:
                return google_search(query, num_results, api_key, search_engine_id)
            except Exception as e:
                print(f"Google search failed: {e}")
    
    # Fallback to Baidu for Chinese users
    print("Trying Baidu search...")
    baidu_results = baidu_search(query, num_results)
    if baidu_results:
        return baidu_results
    
    # Final fallback to local suggestions
    print("Using local search fallback...")
    return local_search_fallback(query, num_results)

def google_search(query: str, num_results: int = 5, api_key: str = None, search_engine_id: str = None) -> List[Dict]:
    """
    Perform a web search using Google Custom Search API.
    
    Args:
        query: The search query string
        num_results: Number of results to return (max 10 per request)
        api_key: Google Custom Search API key
        search_engine_id: Google Custom Search Engine ID
        
    Returns:
        List of search results with title, url, snippet
    """
    if not api_key:
        api_key = os.getenv('GOOGLE_API_KEY')
    if not search_engine_id:
        search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    if not api_key or not search_engine_id:
        raise ValueError("Google API key and Search Engine ID are required. "
                        "Set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables, "
                        "or pass them as parameters.")
    
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': query,
        'num': min(num_results, 10)  # Google allows max 10 results per request
    }
    
    response = requests.get(base_url, params=params, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    results = []
    
    if 'items' in data:
        for item in data['items']:
            result = {
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'display_link': item.get('displayLink', ''),
                'formatted_url': item.get('formattedUrl', ''),
                'source': 'google'
            }
            results.append(result)
    
    return results

def deep_search_enhance(query: str, context: str = "", num_results: int = 3, 
                       api_key: str = None, search_engine_id: str = None) -> Dict:
    """
    Perform deep web search to enhance video analysis.
    
    Args:
        query: Search query based on video content
        context: Additional context from video analysis
        num_results: Number of search results to fetch
        api_key: Google Custom Search API key
        search_engine_id: Google Custom Search Engine ID
        
    Returns:
        Dictionary containing search results
    """
    # Combine query with context
    full_query = f"{query} {context}".strip()
    
    # Perform search
    search_results = web_search(full_query, num_results, api_key, search_engine_id)
    
    return {
        "search_query": full_query,
        "search_results": search_results,
        "note": "Search results from Google Custom Search API"
    }