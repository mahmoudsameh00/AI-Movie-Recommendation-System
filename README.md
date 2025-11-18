# 🎬 Hybrid AI Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green)

A robust, content-based recommendation engine that leverages a **Hybrid Neural Architecture**. It fuses **Deep Autoencoders** (for tabular metadata) and **SBERT** (for natural language plots) to generate rich feature vectors. These vectors are indexed using **FAISS** for millisecond-level similarity searches, wrapped in a dynamic Streamlit interface.

## 🌟 Key Features

* **Hybrid Embedding Strategy:**
    * **Tabular Data:** Compresses high-dimensional features (One-Hot Encoded Genres/Languages, Runtime, Popularity, Weighted Ratings) into dense vectors using a custom **Keras Autoencoder**.
    * **Text Data:** Semantic understanding of movie overviews using the `all-MiniLM-L6-v2` **Sentence Transformer**.
* **High-Performance Search:** Utilizes **Facebook AI Similarity Search (FAISS)** for ultra-fast nearest neighbor retrieval.
* **Cold Start Solved:** If a movie isn't in the dataset, the system fetches it live via the **TMDB API**, processes it on-the-fly, and generates recommendations instantly.
* **Continuous Learning:** New user queries are automatically vectorized and added to the database (saved to disk in the background), constantly expanding the dataset.
* **Self-Healing Data:** Automatically spots missing or invalid details in recommendation outputs and fills them in by **TMDB API**, using the release year to make sure it grabs the exact right movie.

  
## 🛠️ Architecture

### Phase 1: Offline Training Pipeline
The data ingestion and model training process involves cleaning the dataset, training the Autoencoder for tabular data, generating SBERT embeddings for text, and building the FAISS index.

![Offline Training Pipeline](docs/pipeline_phase1.jpeg)

1.  **Data Ingestion:** Raw metadata is cleaned and feature-engineered to produce a training set. JSON artifacts map categorical features.
2.  **Hybrid Vectorization:**
    * **Text Branch:** Movie Plots $\rightarrow$ SBERT $\rightarrow$ 384-dim vector.
    * **Tabular Branch:** Metadata $\rightarrow$ Autoencoder $\rightarrow$ 32-dim vector.
    * **Fusion:** Concatenated into a **416-dim Hybrid Embedding**.
3.  **Indexing:** Vectors are normalized and stored in `prebuilt_index.faiss`. Metadata is cached in `movie_details.json`.

### Phase 2: Live Streamlit Application
The real-time inference logic handles user input, checks the database, performs "Smart Scraping" for missing data, and retrieves recommendations via FAISS.

![Live App Flowchart](docs/pipeline_phase2.jpeg)

1.  **User Input:** The system checks if the movie exists in the internal database.
2.  **Cold Start Handling:** If missing, `Scrap2.py` fetches live data, `Dummies.py` handles alignment, and the pipeline generates a new embedding on the fly.
3.  **Retrieval:** The query vector searches the FAISS index for the top $k$ nearest neighbors.
4.  **Self-Healing:** If display details are incomplete, `Scrap2.py` performs a secondary fetch (using year hints) to patch the data before display.

## 📂 Project Structure

```text
.
├── app.py                   # Main Streamlit application entry point
├── data/                    # Data storage
│   ├── embeddings.csv       # Final vector embeddings
│   ├── genre_columns.json   # Serialized genre features
│   ├── language_columns.json# Serialized language features
│   ├── movie_details.json   # Lookup for posters/ratings (Frontend cache)
│   └── my_clean_data.csv    # Cleaned dataset
├── models/                  # Trained artifacts
│   ├── encoder_model_last.keras  # The Encoder half of the Autoencoder
│   ├── prebuilt_index.faiss      # The FAISS vector index
│   └── preprocessor.pkl          # Sklearn ColumnTransformer
├── notebooks/               # Experiments & Training
│   ├── data_preparation.ipynb    # ETL: Cleaning, filtering, and feature engineering
│   └── model_training.ipynb      # Training Autoencoder & creating FAISS index
├── raw_data/                # Input storage
│   ├── my_clean_data.csv    # Cleaned dataset (Output of data_preparation)
│   ├── dataframe_backup.csv # Cleaned metadata (Title, Poster, Year, Rate) used to build JSON
│   └── movies_metadata.csv  # Source dataset (Required)
└── scripts/                 # Helper Modules
    ├── Dummies.py           # Inference logic for One-Hot encoding
    ├── Rate.py              # Implementation of IMDB Weighted Rating Formula
    ├── Scrap2.py            # TMDB API wrapper for fetching movie metadata
    ├── build_assets.py      # Script to generate consistent JSON assets for One-Hot encoding
    └── convert_json.py          # Utility to convert DataFrame to JSON for frontend display
      
```
## 🚀 Installation

1.  **Clone the repo:**
```bash
git clone [https://github.com/mahmoudsameh00/AI-Movie-Recommendation-System.git](https://github.com/mahmoudsameh00/AI-Movie-Recommendation-System.git)
cd movie-recommender
 ```

2.  **Install dependencies:**
 ```bash
 pip install -r requirements.txt
 ```

3.  **Setup API Key:**
 * Get a free API Key from [The Movie DB (TMDB)](https://www.themoviedb.org/) and replace it in scripts/Scrap2.py and app.py.
     ```ini
     API_KEY=your_api_key_here
     ```

4.  **Run the App:**
 ```bash
 streamlit run app.py
 ```

## 🤝 Acknowledgements
* Python: 3.9
* Dataset: [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv)
* API: [TMDB](https://www.themoviedb.org/)
