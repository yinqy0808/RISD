import matplotlib
matplotlib.use("Agg")

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ANKH_BASE_DIR, configure_environment

configure_environment()

import torch
import pandas as pd
from torch.utils import data
from transformers import T5EncoderModel, AutoTokenizer

HYPERAMP_DIR = os.path.dirname(os.path.abspath(__file__))
if HYPERAMP_DIR not in sys.path:
    sys.path.insert(0, HYPERAMP_DIR)

from HyperAMP import HGNN
from utils import build_batched_samples, collate_padding

torch.manual_seed(1234)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_ANKH_PATH = str(ANKH_BASE_DIR)

_DEFAULT_BATCH_SIZE = 128

_GLOBAL_SCORER = None
_GLOBAL_SCORER_KEY = None


class HyperAMPScorer:
    def __init__(self, model_path: str, device: str | None = None, batch_size: int = _DEFAULT_BATCH_SIZE):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model_path = model_path

        self.premodel = T5EncoderModel.from_pretrained(
            LOCAL_ANKH_PATH,
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_ANKH_PATH,
            local_files_only=True,
        )
        self.premodel.eval()
        self.premodel.to(device=self.device)

        self.model = HGNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _predict_loader(self, data_loader):
        preds = torch.Tensor()
        true_labels = torch.Tensor()

        self.model.eval()
        with torch.no_grad():
            for _, (emb, params, label) in enumerate(data_loader):
                emb = emb.to(self.device)
                params = [para.to(self.device) for para in params]
                label = label.to(self.device)

                output = self.model(emb.to(torch.float32), params)
                preds = torch.cat((preds, output.detach().cpu()), 0)
                true_labels = torch.cat((true_labels, label.detach().cpu()), 0)

        return preds.numpy().flatten(), true_labels.numpy().flatten()

    def score_df(self, data_test: pd.DataFrame):
        test_dataset = build_batched_samples(
            data_test,
            self.premodel,
            self.tokenizer,
            self.device,
            embedding_batch_size=self.batch_size,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_padding,
        )
        preds, _ = self._predict_loader(test_loader)
        return preds

    def score_seqs(self, seqs):
        df_data = build_score_df(seqs)
        return self.score_df(df_data)


def build_score_df(seqs):
    target_keys = []
    y = []
    for i, __ in enumerate(seqs):
        target_keys.append(f"seq{i}")
        y.append(0)

    df_data = pd.DataFrame(zip(target_keys, seqs, y))
    df_data.rename(columns={0: "name", 1: "seq", 2: "label"}, inplace=True)
    return df_data


def clear_predictor_cache():
    global _GLOBAL_SCORER, _GLOBAL_SCORER_KEY
    try:
        if _GLOBAL_SCORER is not None:
            try:
                _GLOBAL_SCORER.model.to("cpu")
            except Exception:
                pass
            try:
                _GLOBAL_SCORER.premodel.to("cpu")
            except Exception:
                pass
    finally:
        _GLOBAL_SCORER = None
        _GLOBAL_SCORER_KEY = None


def init_predictor(model_path: str, device: str | None = None, batch_size: int = _DEFAULT_BATCH_SIZE):
    global _GLOBAL_SCORER, _GLOBAL_SCORER_KEY
    key = (model_path, device, batch_size)
    if _GLOBAL_SCORER is None or _GLOBAL_SCORER_KEY != key:
        clear_predictor_cache()
        _GLOBAL_SCORER = HyperAMPScorer(model_path=model_path, device=device, batch_size=batch_size)
        _GLOBAL_SCORER_KEY = key
    return _GLOBAL_SCORER


def predict_scores_from_df(data_test, model_path, device: str | None = None, batch_size: int = _DEFAULT_BATCH_SIZE):
    scorer = init_predictor(model_path=model_path, device=device, batch_size=batch_size)
    return scorer.score_df(data_test)


def test(data_test, model_path):
    return predict_scores_from_df(data_test, model_path)


def read_seqs(seqs):
    df_data = build_score_df(seqs)
    print("The number of peptides: ", len(df_data))
    return df_data


if __name__ == "__main__":
    test_file = os.path.join(BASE_DIR, "data", "predict.txt")
    data_test = read_dataset(test_file)

    model_name = os.path.join(os.path.dirname(__file__), "hyp_best_model_V1")
    preds = predict_scores_from_df(data_test, model_name)
    pd.DataFrame(preds).to_csv("predict_output_ankh.txt", header=False, index=False)
