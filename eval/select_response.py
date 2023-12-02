# %%
from pathlib import Path
import argparse
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


# %%
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--response_path",
        type=str,
        default="/home/t-jingweiyi/Jingwei/TDC/tdc2023-starter-kit/output/generate/gcg_gens_13b.json",
        help="The data path of full generation",
    )
    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=0.9,
        help="The similarity threshold to filter similar responses.",
    )
    parser.add_argument(
        "--output_response_path",
        type=str,
        default="/home/t-jingweiyi/Jingwei/TDC/tdc2023-starter-kit/output/filter/gcg_gens_13b.json",
        help="The output path of filter response",
    )
    parser.add_argument(
        "--output_meta_path",
        type=str,
        default="/home/t-jingweiyi/Jingwei/TDC/tdc2023-starter-kit/output/filter_meta/gcg_gens_13b.json",
        help="The data path of selected response index and max similar response index",
    )
    args, unknown = parser.parse_known_args()

    return args


def cut_refuse_pattern(response):
    test_prefixes = [
        "I'm sorry",
        "Sorry",
        "I apologize",
        "As an",
        "I'm just",
        "I'm an",
        "I cannot",
        "I would advise",
        "it's not appropriate",
        "As a responsible AI",
        "not a good idea",
    ]
    response = response.split("\n")

    for i, r in enumerate(response):
        if any([p in r for p in test_prefixes]):
            response = response[:i]
            break

    response = "\n".join(response).strip()
    return response


def cut_sure_pattern(response):
    sure_prefixes = [
        "Sure",
    ]
    response = response.split("\n")

    for i, r in enumerate(response):
        if any([p in r for p in sure_prefixes]):
            response = response[i + 1 :]
            break

    response = "\n".join(response).strip()
    return response


def filter_similar_responses(responses, filter_threshold, st_model):
    if len(responses) == 0:
        return responses, [], [], []
    responses = [cut_refuse_pattern(r) for r in responses]

    responses = [cut_sure_pattern(r) for r in responses]

    response_embeddings = st_model.encode(responses)

    sim = np.dot(response_embeddings, response_embeddings.T)
    sim = np.tril(sim, k=-1)

    max_sim = np.max(sim, axis=-1)

    sel_indices = np.argwhere(max_sim < filter_threshold).flatten()
    sel_responses = [responses[i] for i in sel_indices]

    sel_response_embeddings = st_model.encode(sel_responses)
    sel_sim = np.dot(response_embeddings, sel_response_embeddings.T)
    sel_sim = np.tril(sel_sim, k=-1)
    most_sim_response_index = np.argmax(sel_sim, axis=-1)
    response_sim = np.max(sel_sim, axis=-1)

    return (
        sel_responses,
        sel_indices.tolist(),
        most_sim_response_index.tolist(),
        response_sim.tolist(),
    )


if __name__ == "__main__":
    args = parse_args()
    response_path = Path(args.response_path)

    with open(response_path, "r") as f:
        all_responses = json.load(f)

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    filtered_responses = {}
    metadata = {}
    filtered_response_num = 0
    for key in tqdm(all_responses):
        (
            sel_responses,
            sel_indices,
            most_sim_response_index,
            response_sim,
        ) = filter_similar_responses(
            all_responses[key], args.filter_threshold, st_model
        )
        filtered_responses[key] = sel_responses
        metadata[key] = {
            "selected_indices": sel_indices,
            "most_sim_response_index": most_sim_response_index,
            "response_sim": response_sim,
        }
        filtered_response_num += len(sel_responses)

    output_response_path = Path(args.output_response_path)
    output_response_path.parent.mkdir(exist_ok=True, parents=True)
    output_meta_path = Path(args.output_meta_path)
    output_meta_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_response_path, "w") as f:
        json.dump(filtered_responses, f)

    with open(output_meta_path, "w") as f:
        json.dump(metadata, f)

    print(f"Finally select {filtered_response_num} for next step!")
