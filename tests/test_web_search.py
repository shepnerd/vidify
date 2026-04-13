#!/usr/bin/env python3
"""
Test script for multi-region web search functionality
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.extensions.skills.web_search import web_search, deep_search_enhance, is_google_accessible

def test_network_detection():
    """Test network connectivity detection"""
    print("Testing network connectivity...")
    google_accessible = is_google_accessible()
    print(f"Google accessible: {google_accessible}")
    return google_accessible

def test_web_search():
    """Test web search functionality with automatic fallback"""
    print("\nTesting web_search function...")
    try:
        results = web_search("video analysis tutorial", num_results=3)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   Source: {result.get('source', 'unknown')}")
            print(f"   URL: {result['url']}")
            print(f"   Snippet: {result['snippet'][:100]}...")
            print()
    except Exception as e:
        print(f"Error in web_search: {e}")

def test_deep_search():
    """Test deep search enhancement"""
    print("\nTesting deep_search_enhance function...")
    try:
        results = deep_search_enhance("machine learning", "video content analysis", num_results=2)
        print(f"Search query: {results['search_query']}")
        print(f"Found {len(results['search_results'])} results")
        print(f"Note: {results['note']}")
        
        # Show first result details
        if results['search_results']:
            first_result = results['search_results'][0]
            print(f"First result: {first_result['title']}")
            print(f"Source: {first_result.get('source', 'unknown')}")
    except Exception as e:
        print(f"Error in deep_search_enhance: {e}")

def test_chinese_search():
    """Test search with Chinese query"""
    print("\nTesting Chinese search query...")
    try:
        results = web_search("视频分析技术", num_results=2)
        print(f"Found {len(results)} results for Chinese query:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   Source: {result.get('source', 'unknown')}")
            print()
    except Exception as e:
        print(f"Error in Chinese search: {e}")

if __name__ == "__main__":
    print("Vidify Multi-Region Web Search Test")
    print("=" * 50)
    
    google_ok = test_network_detection()
    test_web_search()
    test_deep_search()
    test_chinese_search()
    
    print("\nTest completed!")
    if not google_ok:
        print("Note: Google services are not accessible from your network.")
        print("The system automatically fell back to alternative search methods.")
