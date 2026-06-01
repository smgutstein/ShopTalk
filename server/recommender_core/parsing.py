


def extract_bracketed_choice(llm_response):
    start = llm_response.find("<")
    if start == -1:
        return None

    end = llm_response.find(">", start + 1)
    if end == -1:
        return None

    choice = llm_response[start + 1:end].strip()
    return choice or None


def parse_product_choice(llm_response, found_products):
    choice = extract_bracketed_choice(llm_response)
    dive_deeper = choice == "DIVE DEEPER"

    if choice is None or dive_deeper or choice == "WRONG TRACK":
        return None, {}, dive_deeper

    chosen_product = found_products.get(choice, {})
    if not chosen_product:
        return None, {}, False

    return choice, chosen_product, False



def determine_embedding_mode(user_input, image_path):
    """Describe which query modalities are being used for retrieval."""
    has_text = bool(user_input)
    has_image = image_path is not None

    if has_text and has_image:
        return "text_image"
    if has_image:
        return "image"
    if has_text:
        return "text"
    raise ValueError("At least one query modality is required.")