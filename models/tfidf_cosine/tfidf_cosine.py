from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = None
cosine_sim_matrix = None


def load_model():
    """
    Loads product data, builds TF-IDF features,
    and computes cosine similarity matrix.
    """
    global df, cosine_sim_matrix

    base_path = Path(__file__).resolve().parents[2]
    csv_path = base_path / "data" / "cleaned_data" / "cleaned_products.csv"
    df = pd.read_csv(csv_path)

    df["combined_features"] = (
        df["product_name"].fillna('') + " " +
        df["brand_name"].fillna('') + " " +
        df["category_name"].fillna('') + " " +
        df["sub_category_name"].fillna('') + " " +
        df["short_description"].fillna('') + " " +
        df["long_description"].fillna('')
    )

    df["combined_features"] = df["combined_features"].str.lower().str.strip()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_by_id(product_id, k=10):
    """
    Recommends similar products based on a given product ID.
    Args:
        product_id (int): The ID of the product to find recommendations for.
        k (int): The number of similar products to return.
    Returns:
        list[dict]: A list of dictionaries containing the recommended products and their similarity scores.
    """
    if df is None or cosine_sim_matrix is None:
        raise ValueError("Model not loaded. Call load_model() first.")

    index_list = df.index[df["product_id"] == product_id]

    if len(index_list) == 0:
        return []

    index = index_list[0]

    sim_scores = list(enumerate(cosine_sim_matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_matches = sim_scores[1:k + 1]

    results = []
    for i, score in top_matches:
        results.append({
            "product_id": int(df.iloc[i]["product_id"]),
            "product_name": df.iloc[i]["product_name"],
            "brand_name": df.iloc[i]["brand_name"],
            "category_name": df.iloc[i]["category_name"],
            "similarity_score": round(float(score), 4)
        })

    return results

def precision_at_k(product_id, k=5):
    """
    Evaluates the precision of recommendations for a given product ID.
    Args:
        product_id (int): The ID of the product to evaluate.
        k (int): The number of recommendations to consider for precision calculation.
    Returns: 
        float: The precision at K for the given product ID.
        """
    recs = recommend_by_id(product_id, k)

    true_category = df[df["product_id"] == product_id]["category_name"].values[0]

    relevant = 0
    for r in recs:
        if r["category_name"] == true_category:
            relevant += 1

    return relevant / k







if __name__ == "__main__":
    
    load_model()
    print(recommend_by_id(5, k=5))


    # Test precision for one product
    print("Precision@5:", precision_at_k(5, k=5))

    # Average Precision at K across dataset sample
    scores = []
    for pid in df["product_id"].sample(50):
        scores.append(precision_at_k(pid, k=5))

    print("Average Precision@5:", sum(scores) / len(scores))