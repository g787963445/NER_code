
import os

import torch
from torch import nn


from model.zixing_embedding import GlyphEmbedding
from getzixingid import get_zixing_id



class FusionBertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position, glyph, pinyin and token_type embeddings.
    """

    def __init__(self, config):
        super(FusionBertEmbeddings, self).__init__()
        # config_path = os.path.join(config.name_or_path, 'config')
        # font_files = []
        # for file in os.listdir(config_path):
        #     if file.endswith(".npy"):
        #         font_files.append(os.path.join(config_path, file))

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.glyph_embeddings = GlyphEmbedding(embedding_size=128,zixing_out_dim=config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow models variable name and be able to load
        # any TensorFlow checkpoint file
        # self.glyph_map = nn.Linear(1728, config.hidden_size)
        self.map_fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)



    def forward(self, input_ids=None,token_type_ids=None, position_ids=None,inputs_embeds=None,zixing_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # if position_ids is None:
        # position_ids = self.position_ids[:, :seq_length]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # get char embedding, pinyin embedding and glyph embedding
        word_embeddings = inputs_embeds  # [bs,l,hidden_size]
        glyph_embeddings = self.glyph_embeddings(zixing_ids)  # [bs,l,hidden_size]
        # fusion layer
        concat_embeddings = torch.cat((word_embeddings, glyph_embeddings), 2)
        inputs_embeds = self.map_fc(concat_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
