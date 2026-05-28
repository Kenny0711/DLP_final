"""
GRU-based latent context encoder.

Architecture (from proposal slide 12):
    x_{t-k}, ..., x_t  →  GRU  →  hidden  →  Linear  →  tanh  →  h_t

h_t replaces the manually-labeled context z_t used in vanilla LCPO.
"""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """
    Encodes a window of k past observations into a context vector h_t.

    Parameters
    ----------
    obs_dim     : raw observation dimension (e.g. 24 for SUMO intersection)
    gru_hidden  : GRU hidden state size
    context_dim : output context vector size  (h_t dimension)
    """

    def __init__(self, obs_dim: int, gru_hidden: int, context_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.gru_hidden = gru_hidden
        self.context_dim = context_dim

        self.gru  = nn.GRU(obs_dim, gru_hidden, batch_first=True)
        self.proj = nn.Linear(gru_hidden, context_dim)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs_seq : [B, k, obs_dim]

        Returns
        -------
        h_t : [B, context_dim]
        """
        _, h = self.gru(obs_seq)        # h: [1, B, gru_hidden]
        h = h.squeeze(0)                # [B, gru_hidden]
        return torch.tanh(self.proj(h)) # [B, context_dim]


class ObsHistoryEncoder:
    """
    Stateful wrapper around GRUEncoder for online use (no-grad inference).

    Maintains a sliding window of the last k raw observations and returns
    the encoded context h_t = GRUEncoder(window) for the current step.

    Call reset() at every episode boundary.
    """

    def __init__(
        self,
        encoder: GRUEncoder,
        k: int,
        obs_dim: int,
        device: torch.device,
    ):
        self.encoder = encoder
        self.k       = k
        self.obs_dim = obs_dim
        self.device  = device
        self._window: deque = deque(maxlen=k)

    def reset(self) -> None:
        self._window.clear()

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """
        Append obs to history and return h_t.

        Parameters
        ----------
        obs : [obs_dim]

        Returns
        -------
        h_t : [context_dim]
        """
        self._window.append(obs.astype(np.float32))

        # Left-pad with zeros until window is full
        hist: List[np.ndarray] = list(self._window)
        while len(hist) < self.k:
            hist = [np.zeros(self.obs_dim, dtype=np.float32)] + hist

        seq = np.array(hist, dtype=np.float32)[np.newaxis]   # [1, k, obs_dim]
        seq_t = torch.as_tensor(seq, device=self.device)

        with torch.no_grad():
            h = self.encoder(seq_t)   # [1, context_dim]

        return h.squeeze(0).cpu().numpy()   # [context_dim]
