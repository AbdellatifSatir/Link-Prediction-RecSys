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
- **Result:** Successfully built a recommender that handles "sparse" neighborhoods and provides human-readable justifications for its choices.

---

## 🚀 Next Steps (Roadmap)

### Step 1: Graph Representation Learning (Node2Vec)
- Implement random walks to transform nodes into 64D or 128D embeddings.
- Predict links using the Dot Product or Cosine Similarity between user and movie vectors.

### Step 2: Specialized Embeddings (Metapath2Vec)
- Optimize random walks for the bipartite structure (User-Movie-User paths).

### Step 3: Graph Neural Networks (LightGCN)
- Move to **Message Passing** architectures using PyTorch Geometric.
- This represents the "Gold Standard" for modern Link Prediction in RecSys.

---

## 📂 File Structure
- `data/`: Raw MovieLens 100k files.
- `eda.ipynb`: Detailed dataset analysis.
- `graph_construction.ipynb`: NetworkX graph initialization and metadata.
- `heuristic_recommender.ipynb`: Jaccard-based baseline logic.
- `evaluation.ipynb`: "Link Hide" evaluation framework.
- `feature_engineering.ipynb`: Hybrid model combining content and graph data + Explainability testing.
- `download_data.py`: Setup script.
