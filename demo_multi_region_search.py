#!/usr/bin/env python3
"""
Demo script showing multi-region web search capabilities
"""
import sys
sys.path.append('/mnt/shared-storage-gpfs2/sfteval/wy/vidcopilot')

from agent.extensions.skills.web_search import web_search, is_google_accessible

def demo_multi_region_search():
    """Demonstrate multi-region search capabilities"""
    print("🌐 VidCopilot Multi-Region Web Search Demo")
    print("=" * 50)

    # Check network connectivity
    print("1. Network Connectivity Check:")
    google_ok = is_google_accessible()
    print(f"   Google accessible: {'✅ Yes' if google_ok else '❌ No'}")

    if not google_ok:
        print("   📍 Detected: China/Region with Google restrictions")
        print("   🔄 System will use Baidu search or local fallback")
    else:
        print("   🌍 Detected: Global network with Google access")
        print("   🔄 System will prioritize Google Custom Search")

    print("\n2. Testing Search with Automatic Fallback:")

    # Test search
    query = "video analysis techniques"
    print(f"   Searching for: '{query}'")

    try:
        results = web_search(query, num_results=2)
        print(f"   ✅ Found {len(results)} results")

        for i, result in enumerate(results, 1):
            source = result.get('source', 'unknown')
            source_icon = {
                'google': '🔍',
                'baidu': '🇨🇳',
                'local_fallback': '💡'
            }.get(source, '❓')

            print(f"   {i}. {source_icon} [{source}] {result['title'][:60]}...")
            if len(result['snippet']) > 0:
                print(f"      📄 {result['snippet'][:80]}...")

    except Exception as e:
        print(f"   ❌ Search failed: {e}")

    print("\n3. Usage Examples:")
    print("   # For users with Google access:")
    print("   export GOOGLE_API_KEY='your_key'")
    print("   export GOOGLE_SEARCH_ENGINE_ID='your_cx'")
    print("   python agent/main.py --source-type youtube --uri <URL> --mode brief --include-web-search")
    print()
    print("   # For Chinese users (automatic fallback):")
    print("   python agent/main.py --source-type youtube --uri <URL> --mode brief --include-web-search")
    print("   # No extra configuration needed!")

    print("\n4. Fallback Priority:")
    print("   1️⃣ Google Custom Search (if available)")
    print("   2️⃣ Baidu Search (for Chinese users)")
    print("   3️⃣ Local guidance and suggestions")

    print("\n✨ VidCopilot adapts to your network environment automatically!")

if __name__ == "__main__":
    demo_multi_region_search()