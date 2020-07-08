
from dominate.svg import rect
from flask import Flask, render_template, request
from Recommender import Recommender

app = Flask(__name__)
recommender = Recommender()

@app.route('/')
def start_app():
    return render_template("home.html")

@app.route('/form')
def form():
    return render_template("form.html", models=recommender.models)

@app.route('/recommend', methods=['POST'])
def recommend():
    user = request.form["user"]

    return render_template("recommendations.html", history = recommender.getHistory(int(user)),
                                                   recommendations = recommender.recommend(request.form),
                                                   movies = recommender.movieID_to_info )

@app.route("/algorithms")
def algorithms():
    return render_template("algorithms.html", models=recommender.models)

@app.route("/delete/model", methods=['GET'])
def deleteModel():
    algo = request.args["algo"]
    model = request.args["model"]

    return recommender.deleteModel(algo, model)
    


if __name__ == '__main__':
    app.run()