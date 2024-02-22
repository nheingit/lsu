import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy import spatial

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
embeddings_csv_path = os.path.join(current_dir, "processed", "embeddings.csv")
df = pd.read_csv(embeddings_csv_path, index_col=0)
df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def create_context(question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = (
        openai.embeddings.create(input=question, model="text-embedding-ada-002")
        .data[0]
        .embedding
    )

    # Get the distances from the embeddings
    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values("distances", ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row["n_tokens"] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    model="gpt-3.5-turbo",
    question="What is the meaning of life?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=1500,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"I want you to Answer the question based on the context below, and if the question can't be answered based on the context,say 'i don't know'.\".\n\nContext: {context}\n\n---\n\nQuestion: {question}\nSource:\nAnswer:",
                }
            ],
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=0.1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        print(
            f"I want you to Answer the question, and use the context below to help answer the question, and if the question can't be answered based on the context,say 'i don't know'.\".\n\nContext: {context}\n\n---\n\nQuestion: {question}\nSource:\nAnswer:",
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""
