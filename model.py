import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 🔹 Load dataset
def load_data():
    df = pd.read_csv("data/jobs.csv")

    # Convert skills to lowercase and clean spaces
    df['skills'] = df['skills'].apply(lambda x: ", ".join([s.strip().lower() for s in x.split(",")]))

    return df


# 🔹 Train TF-IDF model
def train_model(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['skills'])
    return tfidf, tfidf_matrix


# 🔹 Clean user input
def clean_input(user_skills):
    return ", ".join([s.strip().lower() for s in user_skills.split(",")])


# 🔹 Recommend jobs
def recommend_jobs(user_skills, df, tfidf, tfidf_matrix):

    # Clean input
    user_skills = clean_input(user_skills)

    # Convert to vector
    user_vec = tfidf.transform([user_skills])

    # Compute similarity
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    scores = similarity.flatten()

    # Add similarity column
    df['similarity'] = scores

    # Remove 0 matches
    recommendations = df[df['similarity'] > 0]

    # Sort by highest similarity
    recommendations = recommendations.sort_values(by='similarity', ascending=False)

    # 🔥 Match reason (clean + accurate)
    user_set = set([s.strip() for s in user_skills.split(",")])

    recommendations['match_reason'] = recommendations['skills'].apply(
        lambda x: ", ".join(user_set & set([s.strip() for s in x.split(",")]))
    )

    # Return top 5 results
    return recommendations[['job_role', 'similarity', 'match_reason']].head(5)