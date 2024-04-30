import numpy as np
import time
import torch
import networkx as nx

from pytorch.batch.sparse import make_batch
from pytorch.batch.dense import Batch as D
import pytorch.models

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='FAHM', help='Model for session-based rec.')
    parser.add_argument('--dataset', '-d', type=str, default='IJCAI', help='Benchmarks for session-based rec.')
    parser.add_argument('--validation', action='store_true',
                        help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2048)
    # parser.add_argument('--modes', type=int, default=50)
    # parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    # parser.add_argument('--mode_select_method', type=str, default='random') #
    # parser.add_argument('--no_filters', action='store_true')   ##store_false为true，即不使用Filter；store_true为false，即使用Filter;
    return parser.parse_known_args()[0]

    # configurations initialization
    config_dict = {
        'USER_ID_FIELD': 'session_id',
        'load_col': None,
        # 'neg_sampling': {'uniform':1},
        'neg_sampling': None,
        'benchmark_filename': ['train', 'test'],
        'alias_of_item_id': ['item_id_list'],
        'topk': [5, 10, 20, 50],
        'metrics': ['Recall', 'NDCG', 'MRR','Precision'],
        'valid_metric': 'NDCG@10',
        'eval_args': {
            'mode': 'full',
            'order': 'TO'
        },
        'gpu_id': args.gpu_id,
        "MAX_ITEM_LIST_LENGTH": 200,
        "train_batch_size": 32 if args.dataset == "ijcai_beh" else 64,
        "eval_batch_size": 24 if args.dataset == "ijcai_beh" else 128,
        "customized_eval": 1,
        "enable_hg": 1,
        "simgcl_lambda":0.1,
        # "lmd": 0.1,
        # "sim": 'dot',
        # "tau":1,
        # "no_filters": 0,    #0即使用Filter, 1即不使用Filter
        # "modes":64,
        # "mode_select_method": 'low',  #random; low?
        # "moving_avg":[48],
        # "abaltion": ""
    }



    config = Config(model="FAHM", dataset=f'{args.dataset}', config_dict=config_dict)
    # config['device']="cpu"
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config, log_root="log_test")
    logger = getLogger()

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_dataset, test_dataset = dataset.build()
    train_sampler, test_sampler = create_samplers(config, dataset, [train_dataset, test_dataset])
    if args.validation:
        train_dataset.shuffle()
        new_train_dataset, new_test_dataset = train_dataset.split_by_ratio([1 - args.valid_portion, args.valid_portion])
        train_data = get_dataloader(config, 'train')(config, new_train_dataset, None, shuffle=True)
        test_data = get_dataloader(config, 'test')(config, new_test_dataset, None, shuffle=False)
    else:
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
        test_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training and evaluation
    test_score, test_result = trainer.fit(
        train_data, test_data, saved=True, show_progress=config['show_progress']
    )

    logger.info(set_color('test result', 'yellow') + f': {test_result}')


@torch.no_grad()
def get_batched_data(n, bsize, dim, sparse, seed, device='cuda'):
    tic = time.time()
    adj_list = []
    for _ in range(bsize):
        graph = nx.barabasi_albert_graph(n, 5, seed)
        adj = nx.adjacency_matrix(graph).tocoo()
        adj_list.append(adj)
    print(f'Graph init done in \t{time.time() - tic:.3f}sec')

    tic = time.time()

    # initialize features
    assert dim % 2 == 0
    init_list = []
    for adj in adj_list:
        edge_indices = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long, device=device)  # [2, |E|]
        e = edge_indices.size(1)
        node_feat = torch.randn(n, dim // 2, device=device)  # [N, D/2]
        edge_feat = torch.randn(e, dim // 2, device=device)  # [|E|, D/2]
        init_list.append((edge_indices, node_feat, edge_feat))

    if sparse:
        # get sparse batch
        edge_indices, node_features, edge_features = zip(*init_list)
        batch = make_batch(node_features, edge_indices, edge_features)
    else:
        # get dense batch
        A_list = []
        for edge_indices, node_feat, edge_feat in init_list:
            edge_feat = torch.sparse_coo_tensor(edge_indices, edge_feat, size=(n, n, dim // 2)).to_dense()  # [N, N, D/2]
            node_feat = node_feat[None, ...] * torch.eye(n, device=device)[..., None]  # [N, N, D/2]
            A = torch.cat([node_feat, edge_feat], dim=-1)  # [N, N, D]
            A_list.append(A)
        # setup batch
        A = torch.stack(A_list, dim=0)  # [B, N, N, D]
        n_nodes = [n] * bsize
        batch = D(A, n_nodes)

    print(f'Batch init done in \t{time.time() - tic:.3f}sec')
    return batch


def get_peak_mem_and_reset():
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.peak"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    return peak_bytes_requirement / 1024 ** 3  # unit: GB


def measure(model, G):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = model(G)  # [B, D]
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    forward_t = start.elapsed_time(end) / 1000  # unit: milliseconds

    out = out.sum()

    start.record()
    out.backward()
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    backward_t = start.elapsed_time(end) / 1000  # unit: milliseconds

    peak_mem = get_peak_mem_and_reset()
    return forward_t, backward_t, peak_mem


def main_routine(repeat, n, bsize, n_layers, dim, dim_qk, dim_v, n_heads, dim_ff, readout_dim_qk, readout_dim_v, readout_n_heads):
    print(f'\n\nn = {n}')
    result = {}

    print("DL")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.MLP(2, 0, [2] * n_layers, dim, dim, dim, sparse=False).to('cuda')
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=False, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['dl_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['dl_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['dl_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['dl_forward_t'] = 'OOM'
        result['dl_backward_t'] = 'OOM'
        result['dl_peak_mem'] = 'OOM'
        print(e)

    print("DA")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.Encoder(2, 0, [2] * n_layers, dim, dim, dim, dim_qk, dim_v, dim_ff, n_heads,
                                           readout_dim_qk, readout_dim_v, readout_n_heads, 'default', 'default',
                                           drop_input=0., dropout=0., drop_mu=0., sparse=False).to('cuda')
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=False, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['da_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['da_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['da_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['da_forward_t'] = 'OOM'
        result['da_backward_t'] = 'OOM'
        result['da_peak_mem'] = 'OOM'

    print("DK")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.Encoder(2, 0, [2] * n_layers, dim, dim, dim, dim_qk, dim_v, dim_ff, n_heads,
                                           readout_dim_qk, readout_dim_v, readout_n_heads, 'default', 'generalized_kernel',
                                           drop_input=0., dropout=0., drop_mu=0., sparse=False).to('cuda')
        model.skip_redraw_projections = True
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=False, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['dk_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['dk_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['dk_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['dk_forward_t'] = 'OOM'
        result['dk_backward_t'] = 'OOM'
        result['dk_peak_mem'] = 'OOM'
        print(e)

    print("SL")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.MLP(2, 0, [2] * n_layers, dim, dim, dim, sparse=True).to('cuda')
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=True, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['sl_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['sl_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['sl_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['sl_forward_t'] = 'OOM'
        result['sl_backward_t'] = 'OOM'
        result['sl_peak_mem'] = 'OOM'
        print(e)

    print("SA")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.Encoder(2, 0, [2] * n_layers, dim, dim, dim, dim_qk, dim_v, dim_ff, n_heads,
                                           readout_dim_qk, readout_dim_v, readout_n_heads, 'default', 'default',
                                           drop_input=0., dropout=0., drop_mu=0., sparse=True).to('cuda')
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=True, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['sa_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['sa_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['sa_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['sa_forward_t'] = 'OOM'
        result['sa_backward_t'] = 'OOM'
        result['sa_peak_mem'] = 'OOM'
        print(e)

    print("SK")
    try:
        forward_t, backward_t, peak_mem = [], [], []
        model = hot_pytorch.models.Encoder(2, 0, [2] * n_layers, dim, dim, dim, dim_qk, dim_v, dim_ff, n_heads,
                                           readout_dim_qk, readout_dim_v, readout_n_heads, 'default', 'generalized_kernel',
                                           drop_input=0., dropout=0., drop_mu=0., sparse=True).to('cuda')
        model.skip_redraw_projections = True
        for i in range(repeat):
            G = get_batched_data(n, bsize, dim, sparse=True, seed=i)
            ft, bt, pm = measure(model, G)
            forward_t.append(ft)
            backward_t.append(bt)
            peak_mem.append(pm)
        result['sk_forward_t'] = (np.mean(np.array(forward_t)), np.std(np.array(forward_t)))
        result['sk_backward_t'] = (np.mean(np.array(backward_t)), np.std(np.array(backward_t)))
        result['sk_peak_mem'] = (np.mean(np.array(peak_mem)), np.std(np.array(peak_mem)))
    except RuntimeError as e:
        result['sk_forward_t'] = 'OOM'
        result['sk_backward_t'] = 'OOM'
        result['sk_peak_mem'] = 'OOM'
        print(e)

    return result


def main():
    repeat = 10
    bsize = 1
    n_layers = 4
    dim = 32
    dim_qk = 32
    dim_v = 32
    n_heads = 4
    dim_ff = 32
    readout_dim_qk = 32
    readout_dim_v = 32
    readout_n_heads = 4
    result = {}
    n_list = list((2 ** np.linspace(5, 18, 27, endpoint=True)).astype(int) // 5)  # for log-scale plot
    for n in n_list:
        start = time.time()
        n_result = main_routine(repeat, n, bsize, n_layers, dim, dim_qk, dim_v, n_heads, dim_ff, readout_dim_qk, readout_dim_v, readout_n_heads)
        print(f"{n}: done after {(time.time() - start):.2f} sec")
        print(f"result_{n} = {n_result}")
        result[n] = n_result
        if n_result['sk_forward_t'] == 'OOM':
            break
    print(result)


if __name__ == '__main__':
    main()
