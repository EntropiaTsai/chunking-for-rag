# build_index.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def build_faiss_index(json_path, model_name, index_path):
    # 載入 chunked 文字
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 載入模型
    model = SentenceTransformer(model_name)

    # 計算 embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    # 建立 FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # 儲存 index
    faiss.write_index(index, index_path)

    # 同步儲存 chunks，以便查回原文
    with open(index_path + ".txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")

    print(f"✅ Index 建立完成：{index_path}")

# 範例使用方式
if __name__ == "__main__":
    build_faiss_index(
        json_path="./vector_base/chunked_consecutive.json",
        model_name="all-MiniLM-L6-v2",
        index_path="./faiss_index/consecutive.index"
    )
