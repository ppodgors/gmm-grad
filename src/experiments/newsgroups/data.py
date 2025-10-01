import os
import re
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from tqdm import tqdm
import nltk

def _setup_nltk():
    """Download nltk stopwords if not present."""
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading nltk stopwords...")
        nltk.download('stopwords')

def _clean_text(text):
    """Lowercase, remove non-alphabetic characters and stopwords."""
    _setup_nltk()
    from nltk.corpus import stopwords
    english_stopwords = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in english_stopwords]
    return " ".join(words)

def _vectorize_documents(texts, model):
    """Convert texts to vectors using TF-IDF weighted Word2Vec embeddings."""
    vectorizer = TfidfVectorizer(min_df=6, max_df=0.7)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_dict = {word: i for i, word in enumerate(feature_names)}
    
    vectors = []
    vector_size = model.vector_size
    for i, doc in enumerate(texts):
        word_vectors, weights = [], []
        doc_tfidf_vector = tfidf_matrix[i]
        for word in doc.split():
            if word in model and word in tfidf_dict:
                word_vectors.append(model[word])
                weights.append(doc_tfidf_vector[0, tfidf_dict[word]])
        
        if not word_vectors:
            vectors.append(np.zeros(vector_size))
        else:
            vectors.append(np.average(word_vectors, axis=0, weights=weights))
            
    return np.array(vectors)

def prepare_newsgroups_data(n_worlds, categories, n_per_cat, output_dir):
    """
    Checks if datasets exist. If not, generates and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if all necessary files already exist
    all_files_exist = all(
        os.path.exists(os.path.join(output_dir, f'data_world_{world_id}.pkl'))
        for world_id in range(n_worlds)
    )
            
    if all_files_exist:
        print("All dataset files found. Skipping data generation.")
        return

    print("Dataset files not found. Starting data generation process...")
    
    print("Loading word2vec model (this may take a while)...")
    word_model = api.load('word2vec-google-news-300')
    print("Word2vec model loaded.")

    for world_id in tqdm(range(n_worlds), desc="Creating datasets"):
        file_path = os.path.join(output_dir, f'data_world_{world_id}.pkl')
        if os.path.exists(file_path):
            continue

        np.random.seed(world_id)
        newsgroups_data = fetch_20newsgroups(
            subset='all', categories=categories, shuffle=True, 
            random_state=world_id, remove=('headers', 'footers', 'quotes')
        )
        
        sampled_docs, sampled_labels = [], []
        for cat_id in np.unique(newsgroups_data.target):
            cat_indices = np.where(newsgroups_data.target == cat_id)[0]
            n_to_sample = min(n_per_cat, len(cat_indices))
            sampled_indices = np.random.choice(cat_indices, n_to_sample, replace=False)
            for idx in sampled_indices:
                sampled_docs.append(newsgroups_data.data[idx])
                sampled_labels.append(newsgroups_data.target[idx])
        
        cleaned_docs = [_clean_text(doc) for doc in sampled_docs]
        doc_vectors = _vectorize_documents(cleaned_docs, word_model)
        
        norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_vectors_normalized = doc_vectors / norms
        
        permutation = np.random.permutation(len(doc_vectors_normalized))
        X_final = doc_vectors_normalized[permutation]
        y_final = np.array(sampled_labels)[permutation]
        
        data_to_save = {'vectors': X_final, 'labels': y_final, 'categories': categories, 'world_id': world_id}
        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f)
            
    print(f"\nCreated and saved {n_worlds} datasets in '{output_dir}'")