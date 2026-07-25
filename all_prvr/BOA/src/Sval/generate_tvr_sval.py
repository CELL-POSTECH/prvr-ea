"""Build BOA's TVR SVAL artifact with the repository's original pipeline."""
import json
import os
from pathlib import Path

from sval import (cluster_map, keywords_cluster, keywords_dict_construct,
                  load_glove_model)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT_ROOT = Path(os.environ.get("PRVR_PROJECT_ROOT", Path(__file__).resolve().parents[4]))
DATASET_ROOT = Path(os.environ.get("PRVR_DATA_ROOT", PROJECT_ROOT / "datasets"))
DATA_ROOT = str(DATASET_ROOT / "tvr" / "TextData")
GLOVE_FILE = str(DATASET_ROOT / "glove.840B.300d.txt")
GLOVE_CACHE = os.path.join(REPO_ROOT, "glove", "glove.840B.300d_model.pkl")
OUTPUT = os.path.join(os.path.dirname(__file__), "data", "tvr", "sval.json")
CLUSTER_NUM = 200


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    glove_model = load_glove_model(GLOVE_FILE, GLOVE_CACHE)
    keywords = keywords_dict_construct(
        os.path.join(DATA_ROOT, "tvrtrain.caption.txt"), glove_model)
    v_keywords = keywords_dict_construct(
        os.path.join(DATA_ROOT, "tvrval.caption.txt"), glove_model)
    keywords = keywords_cluster(keywords, glove_model, CLUSTER_NUM)
    v_keywords = cluster_map(keywords, v_keywords, glove_model)
    with open(OUTPUT, "w", encoding="utf-8") as file:
        json.dump({"keywords": keywords, "v_keywords": v_keywords,
                   "cluster_num": CLUSTER_NUM}, file, indent=4)
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
