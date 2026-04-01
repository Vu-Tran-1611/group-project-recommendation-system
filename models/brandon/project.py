import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os
import joblib

def build_model(df):
    if not os.path.exists("recommendation_system"):
        os.mkdir("recommendation_system")

    combined_text = []

    for i in range(len(df)):
        name = str(df.loc[i, "product_name"]) if pd.notna(df.loc[i, "product_name"]) else ""
        brand = str(df.loc[i, "brand_name"]) if pd.notna(df.loc[i, "brand_name"]) else ""
        short_desc = str(df.loc[i, "short_description"]) if pd.notna(df.loc[i, "short_description"]) else ""
        long_desc = str(df.loc[i, "long_description"]) if pd.notna(df.loc[i, "long_description"]) else ""

        text = name + " " + brand + " " + short_desc + " " + long_desc
        combined_text.append(text.lower())

    df["text"] = combined_text

    vectorizer = TfidfVectorizer(stop_words="english")
    feature_matrix = vectorizer.fit_transform(df["text"])

    model = NearestNeighbors(metric="cosine", n_neighbors=11)
    model.fit(feature_matrix)

    joblib.dump(feature_matrix, "recommendation_system/feature_matrix.joblib")
    joblib.dump(model, "recommendation_system/knn_model.joblib")

    return feature_matrix, model

def recommend_by_id(product_id, feature_matrix, model, df, k=10):
    product_index = -1

    for i in range(len(df)):
        if df.loc[i, "product_id"] == product_id:
            product_index = i
            break

    if product_index == -1:
        return []

    distances, indices = model.kneighbors(feature_matrix[product_index], n_neighbors=k + 1)

    recommendations = []

    for i in indices[0]:
        if i != product_index:
            recommendations.append(int(df.loc[i, "product_id"]))

    return recommendations[:k]

def main():
    df = pd.read_csv("cleaned_products.csv")

    feature_matrix, model = build_model(df)

    product_id = 5
    recs = recommend_by_id(product_id, feature_matrix, model, df)

    print("Recommended product IDs:")
    print(recs)

if __name__ == "__main__":
    main()