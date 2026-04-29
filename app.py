from flask import Flask, render_template, request
from model import load_data, train_model, recommend_jobs

app = Flask(__name__)

df = load_data()
tfidf, tfidf_matrix = train_model(df)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    
    if request.method == "POST":
        skills = request.form["skills"]
        results = recommend_jobs(skills, df, tfidf, tfidf_matrix)
    
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)