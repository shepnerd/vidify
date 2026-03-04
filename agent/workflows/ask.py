# agent/workflows/ask.py
import json
from openai import OpenAI
from agent.skills.persist import load_analysis
from agent.skills.rag_faiss import search_faiss

def wf_ask(asset, question: str,
           llm_base_url: str, llm_model: str,
           embed_base_url: str, embed_model: str,
           top_k: int = 5) -> dict:
    analysis = load_analysis(asset.cache_dir)
    rag = (analysis.get("rag") or {}).get("faiss")
    if not rag:
        raise RuntimeError("No FAISS index. Run wf_index first.")
    hits = search_faiss(rag["index_dir"], question, embed_base_url, embed_model, top_k=top_k)

    # vLLM chat 需要 tokenizer 内有 chat template，否则 chat 请求会报错 [1]
    client = OpenAI(base_url=llm_base_url, api_key="EMPTY")
    payload = {"question": question, "chunks": hits}

    resp = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}],
        temperature=0.2,
        max_completion_tokens=800,
        # vLLM 支持 extra_body 透传 top_k 等额外采样参数 [1]
        # extra_body={"top_k": 50},
    )
    text = resp.choices[0].message.content.strip()
    try:
        result = json.loads(text)
    except Exception:
        result = {"answer": text, "evidence": []}
    return {"result": result, "hits": hits}