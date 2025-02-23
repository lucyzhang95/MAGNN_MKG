import argparse
import pathlib
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score

from MAGNN_utils.data_loader import load_preprocessed_data_2metapaths
from MAGNN_utils.model_tools import IndexGenerator, parse_minibatch
from MAGNN_utils.pytorchtools import EarlyStopping
from model import MAGNN_lp_2metapaths_layer

# Params
num_ntype = 3

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

# load node idx
microbe_idx = pd.read_csv("data/sampled/unique_microbes_idx.dat", sep="\t", encoding="utf-8", header=None)
disease_idx = pd.read_csv("data/sampled/unique_diseases_idx.dat", sep="\t", encoding="utf-8", header=None)

num_microbe = np.int16(len(microbe_idx))
num_disease = np.int16(len(microbe_idx))
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
    lr,
    weight_decay,
    dropout_rate,
    seed,
    do_training=True
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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

        # watch the model to track gradients
        wandb.watch(net, log="all")

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

        # initialize global step counter
        step = 0

        for epoch in range(num_epochs):
            t_start = time.time()
            epoch_train_loss = []

            # training
            net.train()
            total_iterations = train_pos_idx_generator.num_iterations()

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

                epoch_train_loss.append(train_loss.item())

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

                    # sync the step count for iteration and epoch
                    step = epoch * total_iterations + iteration

                    # Log the training loss to wandb
                    wandb.log({"train_loss_per_100_iterations": train_loss.item()}, step=step)

            mean_epoch_loss = np.mean(epoch_train_loss)
            # print epoch training info
            print(f"Epoch {epoch} done: mean train loss = {mean_epoch_loss:.4f}")
            # log last train loss per epoch
            wandb.log({"train_loss_epoch": mean_epoch_loss}, step=step)

            # validation
            net.eval()
            val_loss = []
            pos_proba_list = []
            neg_proba_list = []
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

                    # calculate logits for positive and negative samples
                    pos_out = torch.bmm(pos_embedding_microbe, pos_embedding_disease).squeeze(-1)
                    neg_out = -torch.bmm(neg_embedding_microbe, neg_embedding_disease).squeeze(-1)

                    # calculate validation loss
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))

                    # calculate probabilities and append
                    pos_proba_list.append(torch.sigmoid(pos_out).view(-1))
                    # negative samples are appended with negative sign
                    neg_proba_list.append(torch.sigmoid(-neg_out).view(-1))

            # calculate epoch validation loss
            val_loss = torch.mean(torch.tensor(val_loss))
            # concatenate probabilities
            y_proba_val = torch.cat(pos_proba_list + neg_proba_list).cpu().numpy()

            # construct ground truth labels
            num_pos_samples = sum(p.shape[0] for p in pos_proba_list)
            num_neg_samples = sum(n.shape[0] for n in neg_proba_list)
            y_true_val = np.concatenate([np.ones(num_pos_samples), np.zeros(num_neg_samples)])

            # compute AUC and AP
            val_auc = roc_auc_score(y_true_val, y_proba_val)
            val_ap = average_precision_score(y_true_val, y_proba_val)

            t_end = time.time()
            # print validation info
            print(
                "#Validation: Epoch {:05d} | Val_Loss {:.4f} | Val_AUC {:.4f} | Val_AP {:.4f} | Time(s) {:.4f}".format(
                    epoch, val_loss, val_auc, val_ap, t_end - t_start
                )
            )
            # log the validation loss to wandb
            wandb.log(
                {"val_loss_epoch": val_loss, "val_auc_epoch": val_auc, "val_ap_epoch": val_ap},
                step=step,
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


        # test
        net.eval()
        pos_proba_list = []
        neg_proba_list = []

        cumulative_microbe_embeddings = []
        cumulative_disease_embeddings = []

        all_test_microbes = np.concatenate([
            test_pos_microbe_disease[:, 0],  # microbe column = 0
            test_neg_microbe_disease[:, 0],
        ])
        all_test_diseases = np.concatenate([
            test_pos_microbe_disease[:, 1],  # disease column = 1
            test_neg_microbe_disease[:, 1],
        ])

        unique_microbes = np.unique(all_test_microbes)
        unique_diseases = np.unique(all_test_diseases)

        # create index maps: microbe_id -> row index, disease_id -> col index
        microbe_id_to_row = {m_id: i for i, m_id in enumerate(unique_microbes)}
        disease_id_to_col = {d_id: j for j, d_id in enumerate(unique_diseases)}

        # matrix for all test microbes/diseases, initialize with 0
        test_scores_matrix = np.zeros(
            (len(unique_microbes), len(unique_diseases)),
            dtype=np.float32
        )

        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_microbe_disease_batch = test_pos_microbe_disease[test_idx_batch].tolist()
                test_neg_microbe_disease_batch = test_neg_microbe_disease[test_idx_batch].tolist()

                (
                    test_pos_g_lists,
                    test_pos_indices_lists,
                    test_pos_idx_batch_mapped_lists,
                ) = parse_minibatch(
                    adjlists_microdis,
                    edge_metapath_indices_list_microdis,
                    test_pos_microbe_disease_batch,
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
                    test_neg_microbe_disease_batch,
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

                pairwise_score = torch.sigmoid(
                    torch.matmul(pos_embedding_microbe, pos_embedding_disease.T)
                )
                B = len(test_pos_microbe_disease_batch)  # batch size
                for i in range(B):
                    real_m = test_pos_microbe_disease_batch[i][0]  # microbe ID
                    row_idx = microbe_id_to_row[real_m]
                    for j in range(B):
                        real_d = test_pos_microbe_disease_batch[j][1]  # disease ID
                        col_idx = disease_id_to_col[real_d]

                        # store pairwise score[i,j] in the big matrix
                        test_scores_matrix[row_idx, col_idx] = pairwise_score[i, j].item()

                cumulative_microbe_embeddings.append(pos_embedding_microbe.detach().cpu().numpy())
                cumulative_disease_embeddings.append(pos_embedding_disease.detach().cpu().numpy())

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

                # dot product of microbe and disease matrix embeddings
                pos_out = torch.bmm(pos_embedding_microbe, pos_embedding_disease).flatten()
                neg_out = torch.bmm(neg_embedding_microbe, neg_embedding_disease).flatten()

                # probability of microbe-disease for both positive and negative interactions
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))

            y_proba_test = torch.cat(pos_proba_list + neg_proba_list).cpu().numpy()

        # overall evaluation metrics
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)

        print("Link Prediction Test")
        print("AUC = {}".format(auc))
        print("AP = {}".format(ap))

        # log final test metrics to wandb
        wandb.log({"test_auc": auc, "test_ap": ap})

        auc_list.append(auc)
        ap_list.append(ap)

        row_headers = [f"Microbe_{m}" for m in unique_microbes]
        col_headers = [f"Disease_{d}" for d in unique_diseases]

        pairwise_df = pd.DataFrame(test_scores_matrix, index=row_headers, columns=col_headers)
        pairwise_df.to_csv(f"embeddings/mid_pairwise_scores_{save_postfix}.csv")

        print("Saved pairwise scores matrix with real microbe/disease IDs:", test_scores_matrix.shape)

        final_microbe_embeddings = np.concatenate(cumulative_microbe_embeddings, axis=0)
        final_disease_embeddings = np.concatenate(cumulative_disease_embeddings, axis=0)

        np.save(f"embeddings/mid_microbe_embeddings_{save_postfix}.npy", final_microbe_embeddings)
        np.save(f"embeddings/mid_disease_embeddings_{save_postfix}.npy", final_disease_embeddings)
        print("Saved test embeddings for microbes and diseases.")


    print("----------------------------------------------------------------")
    print("Link Prediction Tests Summary")
    print("AUC_mean = {}, AUC_std = {}".format(np.mean(auc_list), np.std(auc_list)))
    print("AP_mean = {}, AP_std = {}".format(np.mean(ap_list), np.std(ap_list)))


ap = argparse.ArgumentParser(description="MAGNN testing run for sampled MKG dataset")
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
    default=10,
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
    default="MKG_MID",
    help="Postfix for the saved model and result. Default is MKG_MID.",
)
ap.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="Learning rate. Default is 0.005.",
)
ap.add_argument(
    "--weight_decay",
    type=float,
    default=0.001,
)
ap.add_argument(
    "--dropout_rate",
    type=float,
    default=0.5,
    help="Dropout rate. Default is 0.5.",
)
ap.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed. Default is 42.",
)

args = ap.parse_args()


def train():
    wandb.init()

    config = wandb.config

    save_postfix = f"MID_prediction_model_{args.save_postfix}_drp{config.dropout_rate}_seed{config.seed}_wd{config.weight_decay}"

    run_model(
        config.feats_type,
        config.hidden_dim,
        config.num_heads,
        config.attn_vec_dim,
        config.rnn_type,
        config.num_epochs,
        config.patience,
        config.batch_size,
        config.neighbor_samples,
        config.repeat,
        save_postfix,
        config.lr,
        config.weight_decay,
        config.dropout_rate,
        config.seed,
    )


if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "name": "MID lp hyperparameter tuning",
        "metric": {"name": "val_loss_epoch", "goal": "minimize"},
        "parameters": {
            "feats_type": {"values": [0]},
            "hidden_dim": {"values": [32]},
            "num_heads": {"values": [4]},
            "attn_vec_dim": {"values": [32]},
            "rnn_type": {"values": ["RotatE0"]},
            "num_epochs": {"values": [10]},
            "patience": {"values": [5]},
            "batch_size": {"values": [8]},
            "neighbor_samples": {"values": [50]},
            "repeat": {"values": [1]},
            "lr": {"values": [0.0001]},
            "weight_decay": {"values": [0.001, 0.0001]},
            "dropout_rate": {"values": [0.2, 0.3, 0.4, 0.5]},
            "seed": {"values": [42, 10, 62]},
        },
        "early_terminate": {
            "type": "hyperband",
            "max_count": 10
        },
    }

    # create the sweep
    sweep_id = wandb.sweep(sweep_config, project="MAGNN_MKG_LP")

    # start the sweep agent
    wandb.agent(sweep_id, function=train)
