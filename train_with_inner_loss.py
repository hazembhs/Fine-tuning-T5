import os
import argparse
import subprocess
import random
import tempfile
from tqdm import tqdm
import torch
import modeling
import data
import pytrec_eval
from statistics import mean
from collections import defaultdict
import modeling_util

SEED = 42
LR = 0.001
T5_LR = 1e-3
MAX_EPOCH = 20
BATCH_SIZE = 32
BATCH_VALID = 100
BATCHES_PER_EPOCH = 1024
GRAD_ACC_SIZE = 8
VALIDATION_METRIC = 'ndcg_cut_10'
PATIENCE = 5
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

MODEL_MAP = {
    'vanilla_t5': modeling.T5Ranker
}


def main(model, dataset, train_pairs, qrels_train, valid_run, qrels_valid, model_out_dir):
   
    if isinstance(model, str):
        model = MODEL_MAP[model]().cuda()
    if model_out_dir is None:
        model_out_dir = tempfile.mkdtemp()

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_mt5_params = {'params': [v for k, v in params if not k.startswith('t5.')]}
    mt5_params = {'params': [v for k, v in params if k.startswith('t5.')], 'lr': T5_LR}
    optimizer = torch.optim.Adam([non_mt5_params, mt5_params], lr=LR)

    epoch = 0
    top_valid_score = None
    print(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={LR} T5_LR={T5_LR}', flush=True)
    for epoch in range(MAX_EPOCH):

        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels_train)
        print(f'train epoch={epoch} loss={loss}', flush=True)

        valid_score = validate(model, dataset, valid_run, qrels_valid, epoch)
        print(f'validation epoch={epoch} score={valid_score}', flush=True)

        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights', flush=True)
            model.save(os.path.join(model_out_dir, 'weights.p'))
            top_valid_score_epoch = epoch
        if top_valid_score is not None and epoch - top_valid_score_epoch > PATIENCE:
            print(f'no validation improvement since {top_valid_score_epoch}, early stopping', flush=True)
            break

    
    if top_valid_score_epoch != epoch:
        model.load(os.path.join(model_out_dir, 'weights.p'))
    return (model, top_valid_score_epoch)


def train_iteration(model, optimizer, dataset, train_pairs, qrels):
    total = 0
   
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train') as pbar:
        for record in data.iter_train_pairs_with_labels(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
            loss  = model(record['query_tok'],
                                 record['query_mask'],
                                 record['doc_tok'],
                                 record['doc_mask'],
                                 record['labels'])    
            count = len(record['query_id']) // 2           
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss

def validate(model, dataset, run, valid_qrels, epoch):
    run_filtred = {query_id: top_docs for query_id, top_docs in run.items() if any(doc_id in valid_qrels.get(query_id, {}) for doc_id in top_docs)}
    run_scores = run_model(model, dataset, run_filtred)
    rank = {qid: {docid: float(score) for docid, score in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:100]} for qid, docs in run_scores.items()}
    mrr_at_100 = modeling_util.compute_mrr_at_100(rank, valid_qrels)
    print("MRR@100:", mrr_at_100)
    return mrr_at_100

def run_model(model, dataset, run, desc='valid'):
    run_dev = {qid: run[qid] for qid in random.sample(run.keys(), int(len(run) * 0.2))}
    rerank_run = defaultdict(dict)
    true_id = model.tokenizer.get_vocab()[model.tokenizer.tokenize("true")[0]]
    false_id = model.tokenizer.get_vocab()[model.tokenizer.tokenize("false")[0]]
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run_dev.values()), ncols=80, desc=desc) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run_dev, BATCH_VALID):
            logits = model.generate(records['query_tok'],
                                    records['query_mask'],
                                    records['doc_tok'],
                                    records['doc_mask'])  
            false_logits = logits[:, false_id].unsqueeze(dim=-1)
            true_logits = logits[:, true_id].unsqueeze(dim=-1)
            tf_logits = torch.cat((true_logits, false_logits), dim=-1)
            scores = tf_logits.log_softmax(dim=-1)[:, 0]
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run[qid][did] = score.item()
            pbar.update(len(records['query_id']))
    return rerank_run





def main_cli():
    parser = argparse.ArgumentParser('T5 model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_t5')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--qrels_valid', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_t5_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')
    args = parser.parse_args()
    model = MODEL_MAP[args.model]().cuda()
   
    if args.initial_t5_weights is not None:
        model.load(args.initial_t5_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
   
    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    qrels_valid = data.read_qrels_dict(args.qrels_valid)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)

    main(model, dataset, train_pairs, qrels, valid_run, qrels_valid, args.model_out_dir)


if __name__ == '__main__':
    main_cli()