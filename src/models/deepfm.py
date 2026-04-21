import torch
import torch.nn as nn

DENSE_FEATURE_DIM = 13
SPARSE_FEATURE_DIM = 26


class DeepFM(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=8, hidden_units=[128, 64], dropout=0.2):
        super().__init__()

        self.vocab_sizes = vocab_sizes
        self.embed_dim = embed_dim
        self.num_sparse_features = len(vocab_sizes)

        # =========================
        # 1. 一阶线性部分
        # 每个 sparse 特征一个 1 维 embedding
        # =========================
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, 1) for vocab_size in vocab_sizes
        ])

        # dense 特征的一阶线性部分
        self.dense_linear = nn.Linear(DENSE_FEATURE_DIM, 1)

        # =========================
        # 2. FM 二阶交互部分
        # 每个 sparse 特征一个 k 维 embedding
        # =========================
        self.fm_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])

        # =========================
        # 3. Deep 部分
        # 输入 = 所有 sparse embedding 拼接 + dense features
        # =========================
        dnn_input_dim = self.num_sparse_features * embed_dim + DENSE_FEATURE_DIM
        layers = []
        input_dim = dnn_input_dim

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.dnn = nn.Sequential(*layers)
        self.dnn_output = nn.Linear(input_dim, 1)

        # 初始化 embedding
        self._init_weights()

    def _init_weights(self):
        for emb in self.linear_embeddings:
            nn.init.xavier_uniform_(emb.weight)

        for emb in self.fm_embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, dense_x, sparse_x):
        """
        dense_x: [batch_size, 13]
        sparse_x: [batch_size, 26]
        """

        # =========================
        # 1. 一阶线性部分
        # =========================
        linear_sparse_part = []
        for i, emb in enumerate(self.linear_embeddings):
            linear_sparse_part.append(emb(sparse_x[:, i]))  # [batch_size, 1]

        linear_sparse_part = torch.stack(linear_sparse_part, dim=1).sum(dim=1)  # [batch_size, 1]
        linear_dense_part = self.dense_linear(dense_x)  # [batch_size, 1]
        linear_logit = linear_sparse_part + linear_dense_part  # [batch_size, 1]

        # =========================
        # 2. FM 二阶交互部分
        # =========================
        fm_embed_list = []
        for i, emb in enumerate(self.fm_embeddings):
            fm_embed_list.append(emb(sparse_x[:, i]))  # [batch_size, embed_dim]

        fm_embeddings = torch.stack(fm_embed_list, dim=1)  # [batch_size, 26, embed_dim]

        sum_of_embed = torch.sum(fm_embeddings, dim=1)  # [batch_size, embed_dim]
        square_of_sum = sum_of_embed * sum_of_embed

        sum_of_square = torch.sum(fm_embeddings * fm_embeddings, dim=1)
        fm_second_order = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)  # [batch_size, 1]

        # =========================
        # 3. Deep 部分
        # =========================
        dnn_input = torch.cat([
            dense_x,
            fm_embeddings.view(fm_embeddings.size(0), -1)
        ], dim=1)

        dnn_out = self.dnn(dnn_input)
        dnn_logit = self.dnn_output(dnn_out)  # [batch_size, 1]

        # =========================
        # 4. 总输出
        # =========================
        logit = linear_logit + fm_second_order + dnn_logit
        return logit.squeeze(1)