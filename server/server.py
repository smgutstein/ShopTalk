import argparse
import logging

from flask import Flask, jsonify, render_template, request

from recommender import ShopTalkRecommender


parser = argparse.ArgumentParser(description="My Flask Application")
parser.add_argument("-p", "--personality", type=int, default=-1, help="Choose a personality")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
parser.add_argument("-c", "--cpu", action="store_true")
parser.add_argument("-m", "--model", type=str, default="gpt-4o")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

recommender = ShopTalkRecommender(
    personality_index=args.personality,
    debug=args.debug,
    force_cpu=args.cpu,
    model_name=args.model,
)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "template.html",
        conversation_history=recommender.conversation_history,
        personality=recommender.personality,
    )


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_input = data["user_input"]
    return jsonify(recommender.generate_reply(user_input))


if __name__ == "__main__":
    app.run(debug=True)
