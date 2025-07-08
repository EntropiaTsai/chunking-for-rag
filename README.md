# RAG with Three Common Approaches to Text Chunking

```
project-root/
├── original_data.txt.template        # Your custom input text
├── chunking.py                       # Script for three chunking methods
├── build_index.py                    # Builds embeddings + FAISS index
├── RAG.py                            # Query interface for retrieval
├── faiss_index/                      # Folder for saved index and embeddings (empty in repo)
└── vector_base/                      # Folder for saved chunked texts  (empty in repo)

```
This repository demonstrates how different text chunking strategies affect Retrieval-Augmented Generation (RAG) performance. We implement statistical, consecutive, and cumulative semantic chunking, and compare their retrieval behaviours using FAISS and SentenceTransformers.

- Procedure:

    1. Preparing the dataset for retrieval  
    2. Chunking the dataset  
    3. Creating embeddings for each chunk  
    4. Searching similar chunks via `faiss`  

> ⚠️ We highly recommend working in a **virtual environment** to avoid package conflicts.

---

## 1. Preparing the Dataset

Edit the file `original_data.txt.template` by adding the article(s) you want to work with.  
After editing, **rename the file** to `original_data.txt` (or any valid `.txt` filename as referenced in the code).

---

## 2. Chunking the Data

We provide implementations of three common chunking methods in `chunking.py`:  

- **Statistical semantic chunking**  
- **Consecutive semantic chunking**  
- **Cumulative semantic chunking**  

### Install the required package:
```bash
pip install sentence-transformers
```

### Run the script to generate chunked outputs:
```bash
python chunking.py
```

It will output:  
- `chunked_statistical.json`  
- `chunked_consecutive.json`  
- `chunked_cumulative.json`

---

## 3. Creating Embeddings

Run the following to create chunk embeddings and store them using FAISS:

```bash
python build_index.py
```

---

## 4. Searching Similar Chunks via `faiss`

### Install FAISS

🖥️ **MacOS users**:
```bash
pip install faiss-cpu 
```

🪟 **Windows users**:
```bash
pip install faiss
```

### Run the query interface:
```bash
python RAG.py
```

Enter the input text you'd like to query:
```
輸入你想檢索的內容：
```

---


Feel free to explore which chunking strategy works best for your task!
# chunking-for-rag # 建立一個初始 README 檔
# chunking-for-rag
