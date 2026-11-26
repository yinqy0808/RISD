import os
import queue
import sys
import traceback
import uuid
from pathlib import Path
from multiprocessing import get_context

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import get_device

from predict import HyperAMPScorer


def _score_worker_main(in_q, out_q, model_path, device, batch_size):
    scorer = HyperAMPScorer(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
    )

    while True:
        item = in_q.get()
        if item is None:
            break

        req_id, seqs = item
        try:
            scores = scorer.score_seqs(seqs)
            out_q.put((req_id, scores.tolist(), None))
        except Exception:
            out_q.put((req_id, None, traceback.format_exc()))


class ScoreWorkerClient:
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        batch_size: int = 32,
        timeout: int = 600,
        queue_size: int = 8,
    ):
        self.model_path = model_path
        self.device = device or str(get_device())
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue_size = queue_size

        self.ctx = get_context("spawn")
        self.in_q = None
        self.out_q = None
        self.proc = None

    def start(self):
        if self.proc is not None and self.proc.is_alive():
            return

        self.in_q = self.ctx.Queue(maxsize=self.queue_size)
        self.out_q = self.ctx.Queue(maxsize=self.queue_size)
        self.proc = self.ctx.Process(
            target=_score_worker_main,
            args=(self.in_q, self.out_q, self.model_path, self.device, self.batch_size),
            daemon=True,
        )
        self.proc.start()

    def close(self):
        try:
            if self.in_q is not None:
                try:
                    self.in_q.put_nowait(None)
                except Exception:
                    pass
        finally:
            if self.proc is not None:
                self.proc.join(timeout=2)
                if self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(timeout=2)

            self.proc = None
            self.in_q = None
            self.out_q = None

    def restart(self):
        self.close()
        self.start()

    def _ensure_alive(self):
        if self.proc is None or not self.proc.is_alive():
            self.restart()

    def score(self, seqs):
        self._ensure_alive()

        req_id = uuid.uuid4().hex
        try:
            self.in_q.put((req_id, list(seqs)), timeout=self.timeout)
            recv_id, payload, err = self.out_q.get(timeout=self.timeout)
        except Exception:
            self.restart()
            raise RuntimeError("HyperAMP score worker timeout or broken pipe.")

        if recv_id != req_id:
            self.restart()
            raise RuntimeError("HyperAMP score worker response id mismatch.")

        if err is not None:
            self.restart()
            raise RuntimeError(f"HyperAMP score worker failed:\n{err}")

        return np.asarray(payload, dtype=np.float32)
