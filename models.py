
class T5Ranker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mt5 = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
        self.tokenizer =  AutoTokenizer.from_pretrained("google-t5/t5-base")
        self.vocab = self.tokenizer.get_vocab()

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        return toks

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, labels):
        BATCH, QLEN = query_tok.shape
        DIFF = 7  # 6 for query , document and Relevant words and one for SEPS
        maxlen = 512
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)
        
        labels = torch.cat([labels] * sbcount, dim=0)
        
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.eos_token_id)
        
        Query = torch.cat([torch.tensor(self.tokenize('query:')) for _ in range(len(query_toks))], dim=0).view(-1,2).to('cuda:0')
        Doc = torch.cat([torch.tensor(self.tokenize('document:')) for _ in range(len(query_toks))], dim=0).view(-1,2).to('cuda:0')
        Rel = torch.cat([torch.tensor(self.tokenize('relevant:')) for _ in range(len(query_toks))], dim=0).view(-1,2).to('cuda:0')
        
        d_ONES = torch.ones_like(query_mask[:, :2])
        ONES = torch.ones_like(query_mask[:, :1])
       

        toks = torch.cat([Query, query_toks, Doc, doc_toks,Rel ,SEPS], dim=1)
        mask = torch.cat([d_ONES, query_mask, d_ONES, doc_mask,d_ONES ,ONES], dim=1)
        
        labels = torch.cat([labels, SEPS], dim=1)
        
        toks[toks == -1] = 0  

       
        result = self.mt5(input_ids=toks, attention_mask=mask, labels=labels)
        
        
        return result.loss

    def generate(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 7  # 
        maxlen = 512
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)
        
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.eos_token_id)
        
        Query = torch.cat([torch.tensor(self.tokenize('query:')) for _ in range(len(query_toks))], dim=0).view(-1,2).to('cuda:0')
        Doc = torch.cat([torch.tensor(self.tokenize('document:')) for _ in range(len(query_toks))], dim=0).view(-1,2).to('cuda:0')
        Rel = torch.cat([torch.tensor(self.tokenize('relevant:')) for _ in range(len(query_toks))], dim=0).view(-1,2).to('cuda:0')
        
        d_ONES = torch.ones_like(query_mask[:, :2])
        ONES = torch.ones_like(query_mask[:, :1])
       
        toks = torch.cat([Query, query_toks, Doc, doc_toks,Rel ,SEPS], dim=1)
        mask = torch.cat([d_ONES, query_mask, d_ONES, doc_mask,d_ONES ,ONES], dim=1)  
        toks[toks == -1] = 0 
        result = self.mt5.generate(input_ids=toks, attention_mask=mask, output_scores=True,
                                   return_dict_in_generate=True)
        first_score = result.scores[0]
        logits_result = []
        for i in range(first_score.shape[0] // BATCH):
            logits_result.append(first_score[i * BATCH:(i + 1) * BATCH])
        logits_result = torch.stack(logits_result, dim=2).mean(dim=2)
        return logits_result

class T5_with_reranking_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
        self.tokenizer =  AutoTokenizer.from_pretrained("google-t5/t5-base")
        self.vocab = self.tokenizer.get_vocab()

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.encode(text, add_special_tokens=False)
        return toks

    def forward(self, query_tok, query_mask, doc_tok, doc_mask ):
        BATCH, QLEN = query_tok.shape
        DIFF = 5  # 
        maxlen = 512
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)
        
        
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.eos_token_id)
        
        Query = torch.cat([torch.tensor(self.tokenize('query:')) for _ in range(len(query_toks))], dim=0).view(-1,2).to('cuda:0')
        Doc = torch.cat([torch.tensor(self.tokenize('document:')) for _ in range(len(query_toks))], dim=0).view(-1,2).to('cuda:0')
      
        d_ONES = torch.ones_like(query_mask[:, :2])
        ONES = torch.ones_like(query_mask[:, :1])
       

        toks = torch.cat([Query, query_toks, Doc, doc_toks ,SEPS], dim=1)
        mask = torch.cat([d_ONES, query_mask, d_ONES, doc_mask ,ONES], dim=1)

        toks[toks == -1] = 0  # romove padding
        decoder_input_ids = torch.full((32,1), self.tokenizer.pad_token_id)
        result = self.t5(input_ids=toks, attention_mask=mask, decoder_input_ids=decoder_input_ids )
     
        first_logits = result.logits[:, 0]
        
        score = []
        for i in range(first_logits.shape[0] // BATCH):
            score.append(first_logits[i * BATCH:(i + 1) * BATCH])
        score = torch.stack(score, dim=2).mean(dim=2) 
      
        return score

    def generate(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 5  # 
        maxlen = 512
        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF
        
        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)
        doc_toks = doc_toks.to('cuda:0')
        doc_mask = doc_mask.to('cuda:0')
        query_toks = torch.cat([query_tok] * sbcount, dim=0).to('cuda:0')
        query_mask = torch.cat([query_mask] * sbcount, dim=0).to('cuda:0')
        
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.eos_token_id).to('cuda:0')
        
        Query = torch.cat([torch.tensor(self.tokenize('query:')) for _ in range(len(query_toks))], dim=0).view(-1,2).to('cuda:0')
        Doc = torch.cat([torch.tensor(self.tokenize('document:')) for _ in range(len(query_toks))], dim=0).view(-1,2).to('cuda:0')
        
        
        d_ONES = torch.ones_like(query_mask[:, :2]).to('cuda:0')
        ONES = torch.ones_like(query_mask[:, :1]).to('cuda:0')
       
        toks = torch.cat([Query, query_toks, Doc, doc_toks ,SEPS], dim=1).to('cuda:0')
        mask = torch.cat([d_ONES, query_mask, d_ONES, doc_mask ,ONES], dim=1).to('cuda:0')
        toks[toks == -1] = 0 
       
        result = self.t5.generate(input_ids=toks, attention_mask=mask, output_scores=True,
                                   return_dict_in_generate=True)
        
        logits_result = result.scores[0]
        scores = []
        
        for i in range(logits_result.shape[0] // BATCH):
            scores.append(logits_result[i * BATCH:(i + 1) * BATCH])
        scores = torch.stack(scores, dim=2).mean(dim=2)
        return scores
       


