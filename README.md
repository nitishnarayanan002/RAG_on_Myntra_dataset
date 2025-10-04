```markdown
# RAG-Based Fashion Product Search (Myntra Dataset)

This project implements a **Retrieval-Augmented Generation (RAG)** system for Myntra's product catalog. It allows natural-language queries to retrieve and generate detailed product information using embeddings, vector search, caching, and cross-encoder re-ranking.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Objectives](#objectives)  
3. [Dataset](#dataset)  
4. [System Architecture](#system-architecture)  
5. [Tools & Technologies](#tools--technologies)  
6. [Installation](#installation)  
7. [Usage](#usage)  
8. [Caching](#caching)  
9. [Re-ranking](#re-ranking)  
10. [Challenges & Lessons Learned](#challenges--lessons-learned)  
11. [Future Enhancements](#future-enhancements)  

---

## Project Overview

The project enables semantic search over Myntra’s fashion catalog using:

- **Embeddings:** SentenceTransformer models (`all-MiniLM-L6-v2`)  
- **Vector Database:** ChromaDB for storage and search  
- **Caching:** To speed up repeated queries  
- **Re-ranking:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`) for more precise results  

The system supports queries like "Do you have ethnic wear?" or "Describe the fabric and features of W Women."

---

## Objectives

- Enable natural-language search over Myntra products  
- Use embeddings for semantic representation of product documents  
- Store embeddings efficiently in ChromaDB  
- Implement caching for repeated queries  
- Improve result precision using cross-encoder re-ranking  

---

## Dataset

- **Source:** Myntra Product Dataset  
- **Number of products:** 14,215  
- **Columns used:**  
  - `name`, `brand`, `p_attributes`, `colour`, `products`, `price`, `avg_rating`, `p_id`  
- **Document creation:** Combined attributes into a `document_content` column for embeddings  

---

## System Architecture

```

User Query
│
▼
Embedding (SentenceTransformer)
│
▼
Vector Search (ChromaDB)
│
▼
Cross-Encoder Re-Ranking
│
▼
Cache Layer
│
▼
Final Ranked Product Results

````

---

## Tools & Technologies

- **Python 3.10+**  
- **Libraries:** pandas, numpy, sentence-transformers, transformers, chromadb, rank_bm25, cohere, fpdf  
- **Embedding Models:** `all-MiniLM-L6-v2`  
- **Cross-Encoder Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
- **Database:** ChromaDB (Persistent vector database)  
- **Platform:** Google Colab for experimentation  
- **Optional:** Perplexity API for answer generation  

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd <repo_folder>

# Install dependencies
pip install pandas openpyxl langchain pydantic chromadb sentence-transformers transformers accelerate rank_bm25 cohere

# Optional: Upgrade specific packages
pip install --upgrade sentence-transformers
pip install "numpy<2"
````

---

## Usage

```python
# Load your dataset
df = pd.read_excel("Fashion Dataset v2.xlsx")

# Preprocess and create document_content
# (Refer to preprocessing script)

# Initialize embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings and store in ChromaDB
# (Refer to embedding and ChromaDB script)

# Perform cached search
results = get_cached_or_search("Do you have ethnic wear?", k=5)

# Re-rank results
final_results = rerank_documents("Do you have ethnic wear?", results, top_n=3)
```

---

## Caching

* Implemented a **dictionary-based cache** with keys as `(query, k)`
* Reduces repeated computation and speeds up query responses

---

## Re-ranking

* Uses **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`)
* Re-ranks top retrieved documents based on semantic similarity to the query
* Improves brand/attribute-specific accuracy

---

## Challenges & Lessons Learned

**Challenges:**

* High embedding generation time with larger model (`BAAI/bge-small-en-v1.5`) → CPU overload
* Brand mismatch in search results → solved with cross-encoder re-ranking
* Cache invalidation for new queries
* ChromaDB version mismatches

**Lessons Learned:**

* Semantic search + vector DBs outperform keyword search
* Smaller embedding models (`all-MiniLM-L6-v2`) provide a good balance of speed and accuracy
* Re-ranking with cross-encoder significantly improves precision
* Proper preprocessing of product attributes is crucial
* Caching boosts performance for repeated queries

---

## Future Enhancements

* Cloud-hosted vector database for scalability
* LLM-based answer summarization
* Brand/attribute-aware filtering in frontend
* Interactive dashboard for user queries


---

## Author

**Your Name** – [Nitish Narayanan] https://github.com/nitishnarayanan002

```



Do you want me to do that?
```
