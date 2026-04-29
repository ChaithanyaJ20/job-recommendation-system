from model import load_data, train_model, recommend_jobs

def main():
    df = load_data()
    tfidf, tfidf_matrix = train_model(df)
    
    print("Enter your skills (comma separated):")
    user_input = input(">> ")
    
    results = recommend_jobs(user_input, df, tfidf, tfidf_matrix)
    
    print("\nTop Job Recommendations:\n")
    for index, row in results.iterrows():
        print(f"{row['job_role']}  (Score: {round(row['similarity']*100, 2)}%)")

if __name__ == "__main__":
    main()