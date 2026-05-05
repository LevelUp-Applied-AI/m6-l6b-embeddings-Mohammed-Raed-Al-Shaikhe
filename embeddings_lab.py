"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(texts)

    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    
    similarity_matrix = sklearn_cosine(tfidf_matrix)

    return similarity_matrix


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    
    embeddings = {}

    with open(filepath, "r", encoding="utf-8") as file:

        for line in file:

            parts = line.strip().split()

            word = parts[0]

            vector = np.array(parts[1:], dtype=np.float32)

            embeddings[word] = vector

    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    
    words = text.lower().split()

    vectors = []

    for word in words:

        if word in embeddings:

            vectors.append(embeddings[word])

    if len(vectors) == 0:

        return np.zeros(50, dtype=np.float32)

    return np.mean(vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Returns a numpy array of shape (768,).
    """
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"]

    # Expand mask shape for broadcasting
    mask = attention_mask.unsqueeze(-1)

    # Zero out padding tokens
    masked_embeddings = last_hidden_state * mask

    # Sum token embeddings
    summed = masked_embeddings.sum(dim=1)

    # Count valid tokens
    counts = mask.sum(dim=1)

    # Mean pooling
    mean_pooled = summed / counts

    return mean_pooled.squeeze().numpy()


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                        bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself. Return:

        {query_text: {"tfidf": [(text, score), ...],
                    "glove": [(text, score), ...],
                    "bert":  [(text, score), ...]}}
    """
    
    results = {}

    # Build corpus GloVe embeddings
    corpus_glove_embeddings = np.array([
        text_to_glove(text, glove_embeddings)
        for text in texts
    ])

    # Build corpus BERT embeddings
    corpus_bert_embeddings = np.array([
        extract_bert_embedding(text, bert_tokenizer, bert_model)
        for text in texts
    ])

    for query in queries:

        query_idx = texts.index(query)

        # TF-IDF
        tfidf_scores = tfidf_sim[query_idx]

        tfidf_top_indices = np.argsort(tfidf_scores)[::-1]

        tfidf_top_indices = [
            i for i in tfidf_top_indices
            if i != query_idx
        ][:3]

        tfidf_results = [
            (texts[i], float(tfidf_scores[i]))
            for i in tfidf_top_indices
        ]

        # GloVe
        query_glove = text_to_glove(query, glove_embeddings)

        glove_scores = sklearn_cosine(
            [query_glove],
            corpus_glove_embeddings
        )[0]

        glove_top_indices = np.argsort(glove_scores)[::-1]

        glove_top_indices = [
            i for i in glove_top_indices
            if i != query_idx
        ][:3]

        glove_results = [
            (texts[i], float(glove_scores[i]))
            for i in glove_top_indices
        ]

        # BERT
        query_bert = extract_bert_embedding(
            query,
            bert_tokenizer,
            bert_model
        )

        bert_scores = sklearn_cosine(
            [query_bert],
            corpus_bert_embeddings
        )[0]

        bert_top_indices = np.argsort(bert_scores)[::-1]

        bert_top_indices = [
            i for i in bert_top_indices
            if i != query_idx
        ][:3]

        bert_results = [
            (texts[i], float(bert_scores[i]))
            for i in bert_top_indices
        ]

        # Store results
        results[query] = {
            "tfidf": tfidf_results,
            "glove": glove_results,
            "bert": bert_results
        }

    return results


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Load data
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")

    # Task 1: TF-IDF
    result = build_tfidf(texts)
    if result:
        tfidf_matrix, vectorizer = result
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
        if tfidf_sim is not None:
            print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # Task 2: GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} GloVe vectors")
        sample_emb = text_to_glove(texts[0], glove)
        if sample_emb is not None:
            print(f"Sample GloVe text embedding shape: {sample_emb.shape}")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    if sample_bert is not None:
        print(f"Sample BERT embedding shape: {sample_bert.shape}")

    # Task 4: Compare — pick one query per category so the cross-method
    # ranking comparison is not degenerate (the CSV is sorted by category,
    # so texts[:5] would all be from the same one).
    if result and glove and tfidf_sim is not None:
        queries = [df[df["category"] == cat]["text"].iloc[0]
                    for cat in df["category"].unique()]
        comparison = compare_similarities(
            texts, queries, tfidf_sim, glove, model, tokenizer
        )
        if comparison:
            for q in list(comparison.keys())[:2]:
                print(f"\nQuery: {q[:80]}...")
                for method in ["tfidf", "glove", "bert"]:
                    top = comparison[q].get(method, [])
                    print(f"  {method}: {[t[:40] for t, _ in top[:3]]}")
