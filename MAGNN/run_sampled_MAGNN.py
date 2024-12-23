import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from MAGNN_utils.data_loader import load_preprocessed_data_2metapaths
from MAGNN_utils.model_tools import IndexGenerator, parse_minibatch
from MAGNN_utils.pytorchtools import EarlyStopping
from model import MAGNN_lp_2metapaths_layer

# Params
num_ntype = 3  # microbe, disease, metabolite = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001

# [0, 1, 0]: ([0, 1] is 0 and [1, 0] is 1 = [0, 1])
etypes_lists = [
    [[0, 1], [0, 2, 3, 1], [4, 5], [4, 3, 2, 5]],
    [[1, 0], [1, 4, 5, 0], [2, 5, 4, 3], [2, 3]],
]

# any direct relationship between microbe and disease counts as True
use_masks = [
    [True, True, False, False],
    [True, True, False, False],
]
no_masks = [[False] * 4, [False] * 4]
num_microbe = 7180
num_disease = 771
expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 2, 0), (0, 2, 1, 2, 0)],
    [(1, 0, 1), (1, 0, 2, 0, 1), (1, 2, 0, 2, 1), (1, 2, 1)],
]


def run_model(
    feats_type,
    hidden_dim,
    num_heads,
    attn_vec_dim,
    rnn_type,
    num_epochs,
    patience,
    batch_size,
    neighbor_samples,
    repeat,
    save_postfix,
):
    (
        adjlists_microdis,
        edge_metapath_indices_list_microdis,
        _,
        type_mask,
        train_val_test_pos_microbe_disease,
        train_val_test_neg_microbe_disease,
    ) = load_preprocessed_data_2metapaths()

    # device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    features_list = []
    in_dims = []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(
                torch.sparse_coo_tensor(indices, values, torch.Size([dim, dim])).to(device)
            )
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))

    # TODO: after run the model, need to change the name of train_val_test_pos_microbe_disease["train_pos_user_artist"]
    train_pos_microbe_disease = train_val_test_pos_microbe_disease["train_pos_micro_dis"]
    val_pos_microbe_disease = train_val_test_pos_microbe_disease["val_pos_micro_dis"]
    test_pos_microbe_disease = train_val_test_pos_microbe_disease["test_pos_micro_dis"]
    train_neg_microbe_disease = train_val_test_neg_microbe_disease["train_neg_micro_dis"]
    val_neg_microbe_disease = train_val_test_neg_microbe_disease["val_neg_micro_dis"]
    test_neg_microbe_disease = train_val_test_neg_microbe_disease["test_neg_micro_dis"]
    y_true_test = np.array(
        [1] * len(test_pos_microbe_disease) + [0] * len(test_neg_microbe_disease)
    )

    auc_list = []
    ap_list = []

    for _ in range(repeat):
        net = MAGNN_lp_2metapaths_layer(
            [4, 4],
            6,
            etypes_lists,
            in_dims,
            hidden_dim,
            hidden_dim,
            num_heads,
            attn_vec_dim,
            rnn_type,
            dropout_rate,
        )
        net.to(device)

        # use Adam optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        pathlib.Path("checkpoint/").mkdir(parents=True, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            save_path="checkpoint/checkpoint_{}.pt".format(save_postfix),
        )

        dur1, dur2, dur3 = [], [], []
        train_pos_idx_generator = IndexGenerator(
            batch_size=batch_size, num_data=len(train_pos_microbe_disease)
        )
        val_idx_generator = IndexGenerator(
            batch_size=batch_size,
            num_data=len(val_pos_microbe_disease),
            shuffle=False,
        )

        for epoch in range(num_epochs):
            t_start = time.time()

            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_microbe_disease_batch = train_pos_microbe_disease[
                    train_pos_idx_batch
                ].tolist()
                train_neg_idx_batch = np.random.choice(
                    len(train_neg_microbe_disease), len(train_pos_idx_batch)
                )
                train_neg_idx_batch.sort()
                train_neg_microbe_disease_batch = train_neg_microbe_disease[
                    train_neg_idx_batch
                ].tolist()

                (
                    train_pos_g_lists,
                    train_pos_indices_lists,
                    train_pos_idx_batch_mapped_lists,
                ) = parse_minibatch(
                    adjlists_microdis,
                    edge_metapath_indices_list_microdis,
                    train_pos_microbe_disease_batch,
                    device,
                    neighbor_samples,
                    use_masks,
                    num_microbe,
                )
                (
                    train_neg_g_lists,
                    train_neg_indices_lists,
                    train_neg_idx_batch_mapped_lists,
                ) = parse_minibatch(
                    adjlists_microdis,
                    edge_metapath_indices_list_microdis,
                    train_neg_microbe_disease_batch,
                    device,
                    neighbor_samples,
                    no_masks,
                    num_microbe,
                )

                t1 = time.time()
                dur1.append(t1 - t0)

                # embedding extraction for pos and neg samples
                # with bilinear matching and loss computation
                # using binary cross-entropy loss with logits
                [pos_embedding_microbe, pos_embedding_disease], _ = net(
                    (
                        train_pos_g_lists,
                        features_list,
                        type_mask,
                        train_pos_indices_lists,
                        train_pos_idx_batch_mapped_lists,
                    )
                )
                [neg_embedding_microbe, neg_embedding_disease], _ = net(
                    (
                        train_neg_g_lists,
                        features_list,
                        type_mask,
                        train_neg_indices_lists,
                        train_neg_idx_batch_mapped_lists,
                    )
                )

                pos_embedding_microbe = pos_embedding_microbe.view(
                    -1, 1, pos_embedding_microbe.shape[1]
                )
                pos_embedding_disease = pos_embedding_disease.view(
                    -1, pos_embedding_disease.shape[1], 1
                )
                neg_embedding_microbe = neg_embedding_microbe.view(
                    -1, 1, neg_embedding_microbe.shape[1]
                )
                neg_embedding_disease = neg_embedding_disease.view(
                    -1, neg_embedding_disease.shape[1], 1
                )

                pos_out = torch.bmm(pos_embedding_microbe, pos_embedding_disease)
                neg_out = -torch.bmm(neg_embedding_microbe, neg_embedding_disease)

                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        "*Training: Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}".format(
                            epoch,
                            iteration,
                            train_loss.item(),
                            np.mean(dur1),
                            np.mean(dur2),
                            np.mean(dur3),
                        )
                    )

            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_microbe_disease_batch = val_pos_microbe_disease[val_idx_batch].tolist()
                    val_neg_microbe_disease_batch = val_neg_microbe_disease[val_idx_batch].tolist()

                    (
                        val_pos_g_lists,
                        val_pos_indices_lists,
                        val_pos_idx_batch_mapped_lists,
                    ) = parse_minibatch(
                        adjlists_microdis,
                        edge_metapath_indices_list_microdis,
                        val_pos_microbe_disease_batch,
                        device,
                        neighbor_samples,
                        no_masks,
                        num_microbe,
                    )
                    (
                        val_neg_g_lists,
                        val_neg_indices_lists,
                        val_neg_idx_batch_mapped_lists,
                    ) = parse_minibatch(
                        adjlists_microdis,
                        edge_metapath_indices_list_microdis,
                        val_neg_microbe_disease_batch,
                        device,
                        neighbor_samples,
                        no_masks,
                        num_microbe,
                    )

                    [pos_embedding_microbe, pos_embedding_disease], _ = net(
                        (
                            val_pos_g_lists,
                            features_list,
                            type_mask,
                            val_pos_indices_lists,
                            val_pos_idx_batch_mapped_lists,
                        )
                    )
                    [neg_embedding_microbe, neg_embedding_disease], _ = net(
                        (
                            val_neg_g_lists,
                            features_list,
                            type_mask,
                            val_neg_indices_lists,
                            val_neg_idx_batch_mapped_lists,
                        )
                    )

                    pos_embedding_microbe = pos_embedding_microbe.view(
                        -1, 1, pos_embedding_microbe.shape[1]
                    )
                    pos_embedding_disease = pos_embedding_disease.view(
                        -1, pos_embedding_disease.shape[1], 1
                    )
                    neg_embedding_microbe = neg_embedding_microbe.view(
                        -1, 1, neg_embedding_microbe.shape[1]
                    )
                    neg_embedding_disease = neg_embedding_disease.view(
                        -1, neg_embedding_disease.shape[1], 1
                    )

                    pos_out = torch.bmm(pos_embedding_microbe, pos_embedding_disease)
                    neg_out = -torch.bmm(neg_embedding_microbe, neg_embedding_disease)

                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                val_loss = torch.mean(torch.tensor(val_loss))

            t_end = time.time()

            # print validation info
            print(
                "#Validation: Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}".format(
                    epoch, val_loss.item(), t_end - t_start
                )
            )

            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print("Early stopping!")
                break

        test_idx_generator = IndexGenerator(
            batch_size=batch_size,
            num_data=len(test_pos_microbe_disease),
            shuffle=False,
        )
        net.load_state_dict(torch.load("checkpoint/checkpoint_{}.pt".format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_user_artist_batch = test_pos_microbe_disease[test_idx_batch].tolist()
                test_neg_user_artist_batch = test_neg_microbe_disease[test_idx_batch].tolist()

                (
                    test_pos_g_lists,
                    test_pos_indices_lists,
                    test_pos_idx_batch_mapped_lists,
                ) = parse_minibatch(
                    adjlists_microdis,
                    edge_metapath_indices_list_microdis,
                    test_pos_user_artist_batch,
                    device,
                    neighbor_samples,
                    no_masks,
                    num_microbe,
                )

                (
                    test_neg_g_lists,
                    test_neg_indices_lists,
                    test_neg_idx_batch_mapped_lists,
                ) = parse_minibatch(
                    adjlists_microdis,
                    edge_metapath_indices_list_microdis,
                    test_neg_user_artist_batch,
                    device,
                    neighbor_samples,
                    no_masks,
                    num_microbe,
                )

                [pos_embedding_microbe, pos_embedding_disease], _ = net(
                    (
                        test_pos_g_lists,
                        features_list,
                        type_mask,
                        test_pos_indices_lists,
                        test_pos_idx_batch_mapped_lists,
                    )
                )

                [neg_embedding_microbe, neg_embedding_disease], _ = net(
                    (
                        test_neg_g_lists,
                        features_list,
                        type_mask,
                        test_neg_indices_lists,
                        test_neg_idx_batch_mapped_lists,
                    )
                )

                pos_embedding_microbe = pos_embedding_microbe.view(
                    -1, 1, pos_embedding_microbe.shape[1]
                )
                pos_embedding_disease = pos_embedding_disease.view(
                    -1, pos_embedding_disease.shape[1], 1
                )

                neg_embedding_microbe = neg_embedding_microbe.view(
                    -1, 1, neg_embedding_microbe.shape[1]
                )
                neg_embedding_disease = neg_embedding_disease.view(
                    -1, neg_embedding_disease.shape[1], 1
                )

                pos_out = torch.bmm(pos_embedding_microbe, pos_embedding_disease).flatten()
                neg_out = torch.bmm(neg_embedding_microbe, neg_embedding_disease).flatten()

                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))

            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        print("Link Prediction Test")
        print("AUC = {}".format(auc))
        print("AP = {}".format(ap))
        auc_list.append(auc)
        ap_list.append(ap)

    print("----------------------------------------------------------------")
    print("Link Prediction Tests Summary")
    print("AUC_mean = {}, AUC_std = {}".format(np.mean(auc_list), np.std(auc_list)))
    print("AP_mean = {}, AP_std = {}".format(np.mean(ap_list), np.std(ap_list)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MRGNN testing for the recommendation dataset")
    ap.add_argument(
        "--feats-type",
        type=int,
        default=0,
        help="Type of the node features used. "
        + "0 - all id vectors; "
        + "1 - all zero vector. Default is 0.",
    )
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Dimension of the node hidden state. Default is 64.",
    )
    ap.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of the attention heads. Default is 8.",
    )
    ap.add_argument(
        "--attn-vec-dim",
        type=int,
        default=128,
        help="Dimension of the attention vector. Default is 128.",
    )
    ap.add_argument(
        "--rnn-type",
        default="RotatE0",
        help="Type of the aggregator. Default is RotatE0.",
    )
    ap.add_argument(
        "--epoch",
        type=int,
        default=100,
        help="Number of epochs. Default is 100.",
    )
    ap.add_argument("--patience", type=int, default=5, help="Patience. Default is 5.")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size. Default is 8.")
    ap.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of neighbors sampled. Default is 100.",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the training and testing for N times. Default is 1.",
    )
    ap.add_argument(
        "--save-postfix",
        default="MKG_MicroD",
        help="Postfix for the saved model and result. Default is MKG_MicroD.",
    )

    args = ap.parse_args()
    run_model(
        args.feats_type,
        args.hidden_dim,
        args.num_heads,
        args.attn_vec_dim,
        args.rnn_type,
        args.epoch,
        args.patience,
        args.batch_size,
        args.samples,
        args.repeat,
        args.save_postfix,
    )
