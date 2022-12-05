class zixing_embedding(nn.Module):
    def __init__(self,config,zixing2id):
        super().__init__()
        self.zixing_len = len(zixing2id)
        self.embedding_zixing = nn.Embedding(self.zixing_len,config.hidden_size,padding_idx=config.pad_token_id)

    def forward(self,input_zixing_id):
        zixing_final =self.embedding_zixing(input_zixing_id)
        return zixing_final
