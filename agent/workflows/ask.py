# agent/workflows/ask.py
import json
from openai import OpenAI
from agent.models.vllm_openai_client import make_client
from agent.models.direct_model_loader import make_direct_client
from agent.skills.persist import load_analysis
from agent.skills.rag_faiss import search_faiss

def wf_ask(asset, question: str,
           llm_base_url: str, llm_model: str,
           embed_base_url: str, embed_model: str,
           top_k: int = 5,
           direct_model: bool = False,
           model_path: str = None,
           tokenizer_path: str = None) -> dict:
    analysis = load_analysis(asset.cache_dir)
    rag = (analysis.get("rag") or {}).get("faiss")
    if not rag:
        raise RuntimeError("No FAISS index. Run wf_index first.")
    hits = search_faiss(rag["index_dir"], question, embed_base_url, embed_model, top_k=top_k)

    if direct_model:
        client = make_direct_client(model_path, tokenizer_path)
        text = client.chat_with_images(llm_model, json.dumps({"question": question, "chunks": hits}, ensure_ascii=False), [], max_tokens=800, temperature=0.2)
    else:
        # vLLM chat 需要 tokenizer 内有 chat template，否则 chat 请求会报错 [1]
        client = make_client(llm_base_url)
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