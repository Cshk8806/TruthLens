
from flask import Flask, render_template, request
import joblib
from pathlib import Path

from text_preprocessing import preprocess_text

app = Flask(__name__, template_folder="template")

# Load the trained machine learning model
model_path = Path("Models") / "fake_news_model.pkl"
model = joblib.load(model_path)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    scores = None
    news = ""

    if request.method == "POST":

        news = request.form.get("news", "").strip()

        if news:
            # Apply NLTK preprocessing
            processed_news = preprocess_text(news)

            if processed_news.strip():

                # Predict the news category
                prediction = model.predict([processed_news])[0]

                # Get prediction scores
                probabilities = model.predict_proba(
                    [processed_news]
                )[0]

                scores = {
                    label: round(float(score) * 100, 2)
                    for label, score in zip(
                        model.classes_,
                        probabilities
                    )
                }

    return render_template(
        "index.html",
        prediction=prediction,
        scores=scores,
        news=news
    )


if __name__ == "__main__":
    app.run(debug=True)
