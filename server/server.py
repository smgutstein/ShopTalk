import argparse
import csv
from dotenv import load_dotenv

import faiss
import json
import logging
import numpy as np
import os
import pickle
import random
import torch

# Suppress warnings from torchvision.transforms._functional_video (ImageBind import's fault.)
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from flask import Flask, request, jsonify, render_template
from pathlib import Path
from PIL import Image

from imagebind import data
from imagebind.models.imagebind_model import imagebind_huge
from imagebind.models.imagebind_model import ModalityType

#from langchain.chains import ConversationalRetrievalChain
#from langchain.memory import ConversationBufferMemory
#from langchain.memory import ConversationBufferMemory
#from langchain.schema import Document
from langchain_classic.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
#from langchain_core.runnables import RunnableSequence
#from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai import ChatOpenAI

# Define arguments before creating the app
parser = argparse.ArgumentParser(description="My Flask Application")
parser.add_argument("-p", "--personality", type=int, default=-1, help="Choose a personality")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
parser.add_argument("-c", "--cpu", action="store_true")
parser.add_argument("-m", "--model", type=str, default='gpt-4o')

# Parse arguments after app definition
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

if args.cpu:
    device = "cpu"
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")


logging.info(f"Loading ImageBind model...")
start_time = datetime.now()
ibind_model = imagebind_huge(pretrained=True)
ibind_model.eval() # Run in evaluation mode - disables stuff like dropout (and thus variable size batch norms)
ibind_model.to(device)
stop_time = datetime.now()
delta_time = stop_time - start_time 
minutes, seconds = divmod(delta_time.seconds, 60)
embed_load_time = f"{minutes} minutes, {seconds} seconds"
logging.info(f"{minutes} minutes, {seconds} seconds")


def normalize(vectors):
    norms = torch.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

class QueryEmbedder(Embeddings):
    def embed_documents(self, texts):
        logging.info(f"searching multiple strings: {texts}")
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        logging.info(f"embedding one string: {text}")
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([text], device)
        }
        with torch.no_grad():
            outputs = ibind_model(inputs)
            normalized = normalize(outputs[ModalityType.TEXT])
            logging.info(f"text embedding's shape: {outputs[ModalityType.TEXT].shape}")
            logging.info(f"after normalization.  : {normalized.shape}")
        return normalized.cpu().numpy().flatten()

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
os.environ["OPENAI_API_KEY"] = api_key

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Hack for un-interrupted development: allows duplicate OpenMP runtime initialization

# Load the FAISS index and metadata from disk
def load_vector_db(index_path, blurbs_path, id_map_path):
    index = faiss.read_index(index_path)
    logging.info(f"Vector DB info: # of vectors = {index.ntotal}, dims = {index.d}")

    with open(id_map_path, 'rb') as f:
        id_map = pickle.load(f)
        logging.info(f"Vector DB index->pid map size: {len(id_map)}")

    with open(blurbs_path, 'rb') as f:
        logging.info(f"Reading {blurbs_path}")
        blurbs = json.load(f)
        logging.info(f"Blurbs loaded for {len(blurbs)} products.")

    return index, id_map, blurbs

logging.info(f"Loading Database...")
start_time = datetime.now()
# Load the FAISS index and metadata when the app starts
faiss_index, id_map, blurbs = load_vector_db("faiss_index.bin", 
                                             "EDA/product_blurbs/combined_blurb_dict.json", 
                                             "index_to_product_id.pkl")
stop_time = datetime.now()
delta_time = stop_time - start_time 
minutes, seconds = divmod(delta_time.seconds, 60)
logging.info(f"{minutes} minutes, {seconds} seconds")
db_load_time = f"{minutes} minutes, {seconds} seconds"

# Flask app
app = Flask(__name__)

#Define Personality List
personalities = ["Mae West", "Pentecostal preacher", "1920s flapper", 
                 "1950s beatnik", "1960s hippie", "Puritan preacher",
                "Damon Runyon character", "Shakespearean character", 
                "Dickensian character", "1920s gangster", "1950s greaser",
                "Edward Bulwer-Lytton character"]    

if args.personality == -1 or args.personality >= len(personalities):
    personality = random.choice(personalities)
    logging.info(f"Random Personality: {personality}")
else:
    personality = personalities[args.personality]

logging.info(f" {personality}")
chosen_personality = personality

# Initialize ChatOpenAI
chat_openai = ChatOpenAI(api_key=api_key, model=args.model, temperature=0.1)

sys_msg_str = f"You are a helpful shopping assistant with personality: {personality}. " + \
        "You are helping a user find a product, after gathering enough info to make a strong recommendation. " + \
        "Never recommend a product that isn't first provided to you by a system message, " + \
        "and do not ask the user for product IDs or information - it will be given to you automatically."
conversation_history = [
    SystemMessage(content= sys_msg_str),
    AIMessage(content="What would you like to shop for today?")
]

logging.info(f"Loading Image paths...")
start_time = datetime.now()
image_id_to_path = {} #all imgs, not just filtered
with open('images.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        image_id_to_path[row['image_id']] = row['path']
stop_time = datetime.now()
delta_time = stop_time - start_time 
minutes, seconds = divmod(delta_time.seconds, 60)
logging.info(f"{minutes} minutes, {seconds} seconds")
image_path_load_time = f"{minutes} minutes, {seconds} seconds"

if args.debug and Path("debug.txt").exists():
    Path("debug.txt").unlink()  # Clear the debug file
    with open("debug.txt", "a") as f:
        f.write(f"Embedding Load Time: {embed_load_time}\n")
        f.write(f"DB Load Time: {db_load_time}\n")
        f.write(f"Image Path Load Time: {image_path_load_time}\n")
        f.write(f"Chosen Personality: {chosen_personality}\n")
        f.write(f"\n\n")

# Warning: I don't recommend trying to simplify this code.
def all_img_ids(blurb):
    return ([blurb.get("main_image_id")] if isinstance(blurb.get("main_image_id"), str) else (blurb.get("main_image_id") or [])) + \
           ([blurb.get("other_image_id")] if isinstance(blurb.get("other_image_id"), str) else (blurb.get("other_image_id") or []))

def all_img_paths(blurb, image_id_to_path):
    return [image_id_to_path[img_id] for img_id in all_img_ids(blurb)]

def serialize_convo(conversation_history):
    return [
        {"type": msg.__class__.__name__, "content": msg.content} for msg in conversation_history
    ]

@app.route("/", methods=["GET"])
def index():
    return render_template("template.html", conversation_history=conversation_history, personality=personality)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_input = data["user_input"]
    logging.info(f"\n\n\nUser input: {user_input}")

    global conversation_history
    conversation_history.append(HumanMessage(content=user_input))
    # logging.info(f"conversation_history: {conversation_history}\n\n")

    # Use LLM to formulate caption-like text query
    caption_please = "Based on the current conversation, what sort of product should we search for? " + \
            "Please ignore your personality and limit your answer to a maximum of 10 words - " + \
            "words an automated search system would find useful."
    llm_search_query = chat_openai.invoke(conversation_history + \
                                          [SystemMessage(content=caption_please)]).content
    logging.info(f"LLM's suggested search query: {llm_search_query}")

    start_time = datetime.now()
    embedded_query = QueryEmbedder().embed_query(llm_search_query).flatten().tolist()

    D, I = faiss_index.search(np.array([embedded_query]).astype(np.float32), k=10)

    # De-duplicate products
    found_products = {}
    for idx, score in zip(I[0], D[0]):
        pid = id_map[idx]
        blurb = blurbs[pid]
        if pid not in found_products:
            found_products[pid] = {
                "item_name": blurb["item_name"],
                "score": float(score),
                "image_paths": all_img_paths(blurb, image_id_to_path),
                "product_type": blurb["feature_fields"]["product_type"],
                "llm_str": blurbs[pid]['llm_str']
            }
        else:
            found_products[pid]["image_paths"] = found_products[pid]["image_paths"] + [new_img for new_img in all_img_paths(blurb, image_id_to_path) if new_img not in found_products[pid]["image_paths"]]
            # TODO: debug msg if this causes an actual change to the list of images.


    # Give LLM info about product search results
    # get the text from the results
    source_knowledge = "\n\n;\n\n".join([f"product_id: {pid}, item_name: {info['item_name']}" for pid, info in found_products.items()])
    # logging.info(f"Suggested products:")
    # feed into an augmented prompt
    augmented_prompt = "Your next output must be surrounded by <> symbols, filled according to 1 of the following 3 options, with no trailing period:\n" + \
            "A. If any of the listed >>Suggested Products<< below are relevant and suggestible, " + \
            "please output its product ID (NOT it's product NAME!), like: <B071K17SWD>.\n" + \
            "B. If you think that refining search results won't lead to better results, please output: <WRONG TRACK>.\n" + \
            "C. If you think that search results are promising and there's room to ask the user for more specificity, please output: <DIVE DEEPER>.\n" + \
            ">>Suggested Products<<:" + \
            f"{source_knowledge}"
    # logging.info(f"augmented_prompt: {augmented_prompt}")
    ps = []
    for info in found_products.values():
        ps += [info["item_name"]]
    logging.info(f"VectorDB search results: {ps}")
    conversation_history.append(SystemMessage(augmented_prompt))


    logging.info(f"conversation_history: {conversation_history}\n\n")

    llm_response = chat_openai.invoke(conversation_history).content
    ai_ans = AIMessage(content=llm_response)
    logging.info(f"Initial LLM Response: {llm_response}")

    # Check whether the response contains the expected product ID format
    dive_deeper = "DIVE DEEPER" in llm_response
    if not dive_deeper and "WRONG TRACK" not in llm_response and \
            "<" in llm_response and ">" in llm_response:
        chosen_pid = llm_response.split("<")[1].split(">")[0]
        chosen_product = found_products.get(chosen_pid, None)
    else:
        chosen_pid = None
        chosen_product = {}

    # Delete search result info from conversation
    conversation_history = conversation_history[:-1]

    if chosen_pid:
        # The LLM's response included a formatted product ID, which we should strip from the conversation history.
        AI_ans = AIMessage(content=llm_response)
        reprompt_str = "Let's continue the conversation while recommending the following product (you don't need to describe every detail of the product, just whatever seems relevant for the buyer based on this conversation): "
        reprompt_str += chosen_product["llm_str"]
        reprompt = SystemMessage(content=reprompt_str)

        temp_history = conversation_history + [AI_ans, reprompt]
        llm_response = chat_openai.invoke(temp_history).content
        ai_ans = AIMessage(content=llm_response)
        logging.info(f"No-PID LLM Response: {llm_response}")
    else:
        # Let's make sure the LLM wasn't recommending any fake products.
        AI_ans = AIMessage(content=llm_response)
        if dive_deeper:
            reprompt_str = "Let's continue the conversation so we can find better product matches. " + \
                    "Don't recommend any specific products - we're trying to learn more so we can make better recommendations. " + \
                    "For context, here are the latest top search results, which we find promising and want to be able to dive deeper into:\n" + \
                    f"{source_knowledge}"
            logging.info("Asking the user for more details")
        else:
            reprompt_str = "Let's continue the conversation to see if we can find a search area that's better served by our stock. " + \
                    "Don't recommend any specific products - we're trying to learn more so we can see if we have anything that suits the buyer. " + \
                    "You may want to apologize to them, since we're not finding any relevant products in our searches so far. " + \
                    "For context, here are the latest top search results, which we're finding lacking:\n" + \
                    f"{source_knowledge}"
            logging.info("Redirect the user to another search area")
        reprompt = SystemMessage(content=reprompt_str)

        temp_history = conversation_history + [AI_ans, reprompt]
        llm_response = chat_openai.invoke(temp_history).content
        ai_ans = AIMessage(content=llm_response)
        logging.info(f"No-rec LLM Response: {llm_response}")

    conversation_history += [ai_ans]

    logging.info(f"Chosen pid: {chosen_pid}")
    logging.info(f"Chosen product: {chosen_product}")

    stop_time = datetime.now()
    delta_time = stop_time - start_time 
    minutes, seconds = divmod(delta_time.seconds, 60)
    logging.info(f"Took {minutes} minutes, {seconds} seconds to prepare a response to the user's message.")
    max_score_dict = max(found_products.values(), key=lambda x: x["score"])

    if args.debug:
        with open("debug.txt", "a") as f:
            f.write(f"User input: {user_input}\n")
            f.write(f"  Chosen result: {chosen_result[0]['item_name']}\n")
            f.write(f"  Score: {chosen_result[0]['score']}\n")
            f.write(f"\n")
            f.write(f"  Best Item: {max_score_dict['item_name']}\n")
            f.write(f"  Best Score: {max_score_dict['score']}\n\n")
            f.write(f"  Response Time: {minutes} minutes, {seconds} seconds\n")
            f.write(f"  Response: {llm_response.content}\n")
            f.write(f"\n\n")
            for pid, product in sorted(found_products.items(), key=lambda x: x['score'], reverse=True):
                f.write(f"  {product['item_name']}: {product['score']} \n")
                f.write(f"       {product['image_paths']} \n")
                f.write(f"       {product['llm_str']} \n")
                f.write(f"\n")    
            f.write(f"======================================================\n\n")

    return jsonify({
        "conversation": serialize_convo(conversation_history),
        "chosen_product": chosen_product,
        "personality": personality
    })

if __name__ == "__main__":
    app.run(debug=True)
