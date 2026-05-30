from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.data.interactions import Interactions


class _TwoTowerNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)


class TwoTowerModel:
    """Minimal two-tower retriever trained with a BPR loss over observed pairs.

    A user tower and item tower are plain ``nn.Embedding`` tables of width
    ``dim``. Training uses in-batch negatives: for each observed (user, pos)
    pair in a batch, a negative item is sampled and the BPR objective
    ``-log sigmoid(s_pos - s_neg)`` is minimised. CPU-only, deterministic
    given ``seed``.

    The at-scale run (large dim, more epochs, GPU) lives in
    ``notebooks/03_two_tower_colab.ipynb`` and exports the item/user
    embeddings as ``.npy`` for the pipeline to load.
    """

    def __init__(
        self,
        dim: int = 32,
        epochs: int = 10,
        lr: float = 0.05,
        batch_size: int = 256,
        seed: int = 0,
    ):
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self._net: _TwoTowerNet | None = None
        self.inter: Interactions | None = None
        self.loss_history_: list[float] = []

    def fit(self, inter: Interactions) -> "TwoTowerModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        coo = inter.matrix.tocoo()
        users = torch.as_tensor(coo.row, dtype=torch.long)
        items = torch.as_tensor(coo.col, dtype=torch.long)
        n_pairs = users.shape[0]
        n_items = inter.matrix.shape[1]

        net = _TwoTowerNet(inter.matrix.shape[0], n_items, self.dim)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        net.train()

        self.loss_history_ = []
        for _ in range(self.epochs):
            perm = torch.as_tensor(rng.permutation(n_pairs), dtype=torch.long)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_pairs, self.batch_size):
                idx = perm[start : start + self.batch_size]
                u = users[idx]
                pos = items[idx]
                neg = torch.as_tensor(
                    rng.integers(0, n_items, size=u.shape[0]), dtype=torch.long
                )

                u_vec = net.user_emb(u)
                pos_vec = net.item_emb(pos)
                neg_vec = net.item_emb(neg)
                pos_score = (u_vec * pos_vec).sum(dim=1)
                neg_score = (u_vec * neg_vec).sum(dim=1)
                loss = -nn.functional.logsigmoid(pos_score - neg_score).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            self.loss_history_.append(epoch_loss / max(n_batches, 1))

        net.eval()
        self._net = net
        self.inter = inter
        return self

    def item_embeddings(self) -> np.ndarray:
        assert self._net is not None, "call fit() before item_embeddings()"
        return self._net.item_emb.weight.detach().cpu().numpy()

    def user_embeddings(self) -> np.ndarray:
        assert self._net is not None, "call fit() before user_embeddings()"
        return self._net.user_emb.weight.detach().cpu().numpy()
