# Project: Link Prediction Recommender System (RecSys)

## 🎯 Project Overview
Building a movie recommendation engine using **Link Prediction** concepts. We treat the user-item interaction as a **Bipartite Graph** where recommendations are predicted missing edges.

---

## 🛠️ Current State (Completed)

### Phase 1: Data Preparation & EDA
- **Dataset:** MovieLens 100k.
- **EDA:** Analyzed rating distributions, sparsity (~93.7%), and popularity bias.
- **Decision:** Ratings $\ge 4$ are defined as "Positive Links" (Ground Truth edges).

### Phase 2: Graph Construction
- **Library:** `NetworkX`.
- **Structure:** Bipartite graph ($U, V$) with unique prefixes (`u_` for users, `m_` for movies).
- **Metadata:** Movie titles injected as node attributes for explainability.

### Phase 3: Heuristic Recommender (Baseline)
- **Algorithm:** Collaborative Filtering using **Jaccard Similarity** on the bipartite graph.
- **Logic:** Neighborhood overlap used to identify "peer users" and predict new links.

### Phase 4: Baseline Evaluation
- **Strategy:** "Link Hide" (80% Train / 20% Test split).
- **Metric:** **Precision @ 5** calculated across a sample of test users.

### Phase 5: Feature Engineering & Hybrid Model (Finalized)
- **User Features:** Normalized Age, One-Hot Encoded Gender and Occupation.
- **Movie Features:** Binary Genre flags (19 categories).
- **Hybrid Logic:** Combined Graph (Jaccard) + Content (Cosine) into a weighted similarity score.
- **Explainability:** Implemented `recommend_with_explanations` to distinguish between **Behavioral Peers** (shared taste) and **Demographic Peers** (similar profile).

### Phase 6: Graph Representation Learning (Node2Vec)
- **Algorithm:** DeepWalk-style uniform random walks.
- **Embeddings:** 64D vectors learned using `gensim`'s Word2Vec.
- **Logic:** Recommendations based on cosine similarity in the latent embedding space.

### Phase 7: Heterogeneous Embeddings (Metapath2Vec)
- **Algorithm:** Structured random walks following the **User-Movie-User (UMU)** metapath.
- **Goal:** Capture deep collaborative filtering intent by forcing the walker to alternate between node types.

### Phase 8: Unified Benchmark & Interpretation (Finalized)
- **Goal:** Systematic head-to-head comparison of Jaccard, Hybrid, Node2Vec, and Metapath2Vec.
- **Metrics:** Precision@10, Recall@10, and MRR.
- **Key Finding**: Jaccard (0.149 Precision) currently leads, followed by Hybrid (0.121). Embedding models struggle with sparsity, motivating the move to GNNs.
- **Documentation**: `unified_benchmark.ipynb` updated with detailed cell-by-cell explanations and strategic analysis.

---

## 🚀 Next Steps (Roadmap)

### Step 1: Graph Neural Networks (LightGCN)
- Move to **Message Passing** architectures using PyTorch Geometric.
- This represents the "Gold Standard" for modern Link Prediction in RecSys.

### Step 2: Hyperparameter Optimization
- Fine-tune walk length, window size, and embedding dimensions for Node2Vec and Metapath2Vec to bridge the performance gap with the baseline.

---

## 📂 File Structure
- `data/`: Raw MovieLens 100k files.
- `eda.ipynb`: Detailed dataset analysis.
- `graph_construction.ipynb`: NetworkX graph initialization and metadata.
- `heuristic_recommender.ipynb`: Jaccard-based baseline logic.
- `node2vec_recommender.ipynb`: Node2Vec implementation and T-SNE.
- `metapath2vec_recommender.ipynb`: Metapath2Vec implementation.
- `evaluation.ipynb`: Initial "Link Hide" evaluation framework.
- `feature_engineering.ipynb`: Hybrid model combining content and graph data.
- `unified_benchmark.ipynb`: Comparative analysis and professional report.
- `download_data.py`: Setup script.
