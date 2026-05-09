import argparse
import logging

from flask import Flask, jsonify, render_template, request

from recommender import ShopTalkRecommender


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ShopTalk Flask application.")
    parser.add_argument("-p", "--personality", type=int, default=-1, help="Choose a personality")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-c", "--cpu", action="store_true")
    parser.add_argument("-m", "--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--vector_db_output_dir",
        type=str,
        default="artifacts/vector_db",
        help="Base directory containing generated vector DB artifacts.",
    )
    parser.add_argument(
        "--vector_backend",
        type=str,
        default="faiss",
        choices=["faiss"],
        help="Vector backend to load for serving.",
    )
    parser.add_argument(
        "--product_blurbs",
        type=str,
        default="EDA/product_blurbs/combined_blurb_dict.json",
        help="Path to the product blurbs JSON file.",
    )
    parser.add_argument(
        "--images_csv",
        type=str,
        default="images.csv",
        help="Path to the image ID mapping CSV file.",
    )
    return parser.parse_args()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def build_recommender(args):
    return ShopTalkRecommender(
        personality_index=args.personality,
        debug=args.debug,
        force_cpu=args.cpu,
        model_name=args.model,
        vector_db_output_dir=args.vector_db_output_dir,
        vector_backend=args.vector_backend,
        blurbs_path=args.product_blurbs,
        images_csv_path=args.images_csv,
    )


def create_app(recommender):
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

    return app


def main():
    args = parse_args()
    configure_logging()
    recommender = build_recommender(args)
    app = create_app(recommender)
    app.run(debug=True)


if __name__ == "__main__":
    main()
