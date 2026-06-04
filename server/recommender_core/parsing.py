

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