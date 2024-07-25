import re
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()


model_id = "meta-llama/Prompt-Guard-86M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)


def get_class_probabilities(text, temperature=1.0, device="cpu"):
    """
    Evaluate the model on the given text with temperature-adjusted softmax.
    Note, as this is a DeBERTa model, the input text should have a maximum length of 512.

    Args:
        text (str): The input text to classify.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        torch.Tensor: The probability of each class adjusted by the temperature.
    """
    # Encode the text
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = inputs.to(device)
    # Get logits from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Apply softmax to get probabilities
    probabilities = softmax(scaled_logits, dim=-1)
    return probabilities


def get_jailbreak_score(text, temperature=1.0, device="cpu"):
    """
    Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
    Appropriate for filtering dialogue between a user and an LLM.

    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        float: The probability of the text containing malicious content.
    """
    probabilities = get_class_probabilities(text, temperature, device)
    return probabilities[0, 2].item()


def get_indirect_injection_score(text, temperature=1.0, device="cpu"):
    """
    Evaluate the probability that a given string contains any embedded instructions (malicious or benign).
    Appropriate for filtering third party inputs (e.g. web searches, tool outputs) into an LLM.

    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        float: The combined probability of the text containing malicious or embedded instructions.
    """
    probabilities = get_class_probabilities(text, temperature, device)
    return (probabilities[0, 1] + probabilities[0, 2]).item()


def process_text_batch(texts, temperature=1.0, device="cpu"):
    """
    Process a batch of texts and return their class probabilities.
    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to process.
        temperature (float): The temperature for the softmax function.
        device (str): The device to evaluate the model on.

    Returns:
        torch.Tensor: A tensor containing the class probabilities for each text in the batch.
    """
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    scaled_logits = logits / temperature
    probabilities = softmax(scaled_logits, dim=-1)
    return probabilities


def chunk_text(text, max_chunk_size=450):
    """
    Divide the text into chunks of approximately max_chunk_size tokens,
    preserving sentence structure and potential injections.

    Args:
        text (str): The input text to be chunked.
        max_chunk_size (int): The maximum size of each chunk in tokens (default: 450).

    Returns:
        list: A list of text chunks.
    """
    # Split the text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    for sentence in sentences:
        # Estimate the number of tokens in the sentence
        sentence_size = len(sentence.split())

        if current_chunk_size + sentence_size > max_chunk_size:
            # If adding this sentence exceeds the max chunk size, start a new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_chunk_size = sentence_size
        else:
            # Add the sentence to the current chunk
            current_chunk += " " + sentence
            current_chunk_size += sentence_size

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def get_scores_for_texts(
    texts,
    score_indices,
    temperature=1.0,
    device="cpu",
    max_batch_size=16,
):
    """
    Compute scores for a list of texts, handling texts of arbitrary length by breaking them into chunks and processing in parallel.
    Returns both the highest score for each text and the chunk where the highest score was found.

    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to evaluate.
        score_indices (list[int]): Indices of scores to sum for final score calculation.
        temperature (float): The temperature for the softmax function.
        device (str): The device to evaluate the model on.
        max_batch_size (int): The maximum number of text chunks to process in a single batch.

    Returns:
        list[tuple]: A list of tuples, each containing (highest_score, chunk_with_highest_score) for each input text.
    """
    all_chunks = []
    text_indices = []
    for index, text in enumerate(texts):
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        text_indices.extend([index] * len(chunks))

    all_scores = [0] * len(texts)
    all_issue_texts = [""] * len(texts)

    for i in range(0, len(all_chunks), max_batch_size):
        batch_chunks = all_chunks[i : i + max_batch_size]
        batch_indices = text_indices[i : i + max_batch_size]
        probabilities = process_text_batch(batch_chunks, temperature, device)
        scores = probabilities[:, score_indices].sum(dim=1).tolist()

        for idx, score, chunk in zip(batch_indices, scores, batch_chunks):
            if score > all_scores[idx]:
                all_scores[idx] = score
                all_issue_texts[idx] = chunk

    return list(zip(all_scores, all_issue_texts))


# Update the functions that use get_scores_for_texts
def get_jailbreak_scores_for_texts(
    texts, temperature=1.0, device="cpu", max_batch_size=16
):
    """
    Compute jailbreak scores for a list of texts and return the scores along with the problematic text chunks.

    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to evaluate.
        temperature (float): The temperature for the softmax function.
        device (str): The device to evaluate the model on.
        max_batch_size (int): The maximum number of text chunks to process in a single batch.

    Returns:
        list[tuple]: A list of tuples, each containing (jailbreak_score, chunk_with_highest_score) for each input text.
    """
    return get_scores_for_texts(texts, [2], temperature, device, max_batch_size)


def get_indirect_injection_scores_for_texts(
    texts, temperature=1.0, device="cpu", max_batch_size=16
):
    """
    Compute indirect injection scores for a list of texts and return the scores along with the problematic text chunks.

    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to evaluate.
        temperature (float): The temperature for the softmax function.
        device (str): The device to evaluate the model on.
        max_batch_size (int): The maximum number of text chunks to process in a single batch.

    Returns:
        list[tuple]: A list of tuples, each containing (indirect_injection_score, chunk_with_highest_score) for each input text.
    """
    return get_scores_for_texts(texts, [1, 2], temperature, device, max_batch_size)


def get_combined_injection_scores(
    texts, temperature=1.0, device="cpu", max_batch_size=16
):
    """
    Compute both jailbreak and indirect injection scores for a list of texts and return the scores
    along with the problematic text chunks.

    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to evaluate.
        temperature (float): The temperature for the softmax function.
        device (str): The device to evaluate the model on.
        max_batch_size (int): The maximum number of text chunks to process in a single batch.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'jailbreak_score': The highest jailbreak score for the text.
            - 'jailbreak_text': The chunk of text with the highest jailbreak score.
            - 'indirect_injection_score': The highest indirect injection score for the text.
            - 'indirect_injection_text': The chunk of text with the highest indirect injection score.
            - 'combined_score': The maximum of jailbreak and indirect injection scores.
            - 'combined_text': The chunk of text with the highest combined score.
    """
    all_chunks = []
    text_indices = []
    for index, text in enumerate(texts):
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        text_indices.extend([index] * len(chunks))

    results = [
        {
            "jailbreak_score": 0,
            "jailbreak_text": "",
            "indirect_injection_score": 0,
            "indirect_injection_text": "",
        }
        for _ in texts
    ]

    for i in range(0, len(all_chunks), max_batch_size):
        batch_chunks = all_chunks[i : i + max_batch_size]
        batch_indices = text_indices[i : i + max_batch_size]
        probabilities = process_text_batch(batch_chunks, temperature, device)

        jailbreak_scores = probabilities[:, 2].tolist()
        indirect_injection_scores = (probabilities[:, 1] + probabilities[:, 2]).tolist()

        for idx, jb_score, ii_score, chunk in zip(
            batch_indices, jailbreak_scores, indirect_injection_scores, batch_chunks
        ):
            if jb_score > results[idx]["jailbreak_score"]:
                results[idx]["jailbreak_score"] = jb_score
                results[idx]["jailbreak_text"] = chunk

            if ii_score > results[idx]["indirect_injection_score"]:
                results[idx]["indirect_injection_score"] = ii_score
                results[idx]["indirect_injection_text"] = chunk

    return results
