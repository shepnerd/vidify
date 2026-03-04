# agent/utils/cache.py
import hashlib, os, json

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def write_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0