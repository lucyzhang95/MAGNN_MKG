import re
import pandas as pd
import glob

from sympy.assumptions.lra_satask import lra_satask
from sympy.physics.quantum.identitysearch import lr_op
from torch.nn.functional import leaky_relu_, local_response_norm


def extract_param(file_name, param_prefix):
    match = re.search(rf'{param_prefix}(\d+e-\d+)', file_name)
    return float(match.group(1).replace('e-', 'e-')) if match else None


def extract_data_from_logs_updated(file_path, param_prefix):
    epoch_data = []
    metrics_data = []
    param = extract_param(file_path, param_prefix)

    current_epoch = None
    train_loss = None

    with open(file_path, 'r') as file:
        for line in file:
            train_loss_match = re.search(r"Epoch (\d+) done: mean train loss = ([\d.]+)", line)
            if train_loss_match:
                # convert to 1-based indexing
                current_epoch = int(train_loss_match.group(1)) + 1
                train_loss = float(train_loss_match.group(2))

            val_metrics_match = re.search(
                r"#Validation: Epoch (\d+) \| Val_Loss ([\d.]+) \| Val_AUC ([\d.]+) \| Val_AP ([\d.]+)", line)
            if val_metrics_match:
                epoch = int(val_metrics_match.group(1)) + 1
                val_loss = float(val_metrics_match.group(2))
                val_auc = float(val_metrics_match.group(3))
                val_ap = float(val_metrics_match.group(4))

                if current_epoch == epoch:
                    epoch_data.append({"epoch": epoch, param_prefix: param, "train_loss": train_loss, "val_loss": val_loss})

                metrics_data.append({param_prefix: param, "auc": val_auc, "ap": val_ap})

    return epoch_data, metrics_data


def extract_final_test_metrics(file_path, param_prefix):
    test_metrics_data = []
    param = extract_param(file_path, param_prefix)
    auc_mean = None
    ap_mean = None

    with open(file_path, 'r') as file:
        for line in file:
            test_auc_match = re.search(r"AUC_mean\s*=\s*([\d.]+)", line)
            test_ap_match = re.search(r"AP_mean\s*=\s*([\d.]+)", line)

            if test_auc_match:
                auc_mean = float(test_auc_match.group(1))
            if test_ap_match:
                ap_mean = float(test_ap_match.group(1))

            if auc_mean is not None and ap_mean is not None:
                test_metrics_data.append({param_prefix: param, "auc": auc_mean, "ap": ap_mean})
                break

    return test_metrics_data


if __name__ == "__main__":
    # process lr log txt files for mid prediction
    all_lr_epoch_data = []
    all_metrics_data = []
    file_paths = glob.glob("MID_logs/lr/*.log")
    for file_path in file_paths:
        lr_epoch_data, lr_metrics_data = extract_data_from_logs_updated(file_path, "lr")
        all_lr_epoch_data.extend(lr_epoch_data)
        all_metrics_data.extend(lr_metrics_data)

    lr_epoch_df = pd.DataFrame(all_lr_epoch_data)
    lr_metrics_df = pd.DataFrame(all_metrics_data)
    lr_epoch_csv_path = "R_data/lr_train_eval_log.csv"
    lr_metrics_csv_path = "R_data/lr_val_metrics.csv"
    lr_epoch_df.to_csv(lr_epoch_csv_path, index=False)
    lr_metrics_df.to_csv(lr_metrics_csv_path, index=False)

    all_dp_epoch_data = []
    all_dp_metrics_data = []
    file_paths = glob.glob("MID_logs/dp/*.log")
    for file_path in file_paths:
        dp_epoch_data, dp_metrics_data = extract_data_from_logs_updated(file_path, "dp")
        all_dp_epoch_data.extend(dp_epoch_data)
        all_dp_metrics_data.extend(dp_metrics_data)

    dp_epoch_df = pd.DataFrame(all_dp_epoch_data)
    dp_metrics_df = pd.DataFrame(all_dp_metrics_data)
    dp_epoch_csv_path = "R_data/dp_train_eval_log.csv"
    dp_metrics_csv_path = "R_data/dp_val_metrics.csv"
    dp_epoch_df.to_csv(dp_epoch_csv_path, index=False)
    dp_metrics_df.to_csv(dp_metrics_csv_path, index=False)

    # lr test metrics
    all_lr_final_test_metrics_data = []
    file_paths = glob.glob("MID_logs/lr/*.log")
    for file_path in file_paths:
        final_lr_test_metrics_data = extract_final_test_metrics(file_path, "lr")
        all_lr_final_test_metrics_data.extend(final_lr_test_metrics_data)
    final_lr_test_metrics_df = pd.DataFrame(all_lr_final_test_metrics_data)
    final_lr_test_metrics_csv_path = "R_data/lr_test_metrics.csv"
    final_lr_test_metrics_df.to_csv(final_lr_test_metrics_csv_path, index=False)

    # dp test metrics
    all_dp_final_test_metrics_data = []
    file_paths = glob.glob("MID_logs/dp/*.log")
    for file_path in file_paths:
        final_dp_test_metrics_data = extract_final_test_metrics(file_path, "dp")
        all_dp_final_test_metrics_data.extend(final_dp_test_metrics_data)
    final_dp_test_metrics_df = pd.DataFrame(all_dp_final_test_metrics_data)
    final_dp_test_metrics_csv_path = "R_data/dp_test_metrics.csv"
    final_dp_test_metrics_df.to_csv(final_dp_test_metrics_csv_path, index=False)


    # process lr log txt files for mime prediction
    mime_lr_epoch_data = []
    mime_lr_metrics_data = []
    file_paths = glob.glob("MIME_logs/lr/*.log")
    for file_path in file_paths:
        mime_epoch_data, mime_metrics_data = extract_data_from_logs_updated(file_path, "lr")
        mime_lr_epoch_data.extend(mime_epoch_data)
        mime_lr_metrics_data.extend(mime_metrics_data)

    mime_lr_epoch_df = pd.DataFrame(mime_lr_epoch_data)
    mime_lr_metrics_df = pd.DataFrame(mime_lr_metrics_data)
    mime_lr_epoch_csv_path = "R_data/mime_lr_train_eval_log.csv"
    mime_lr_metrics_csv_path = "R_data/mime_lr_val_metrics.csv"
    mime_lr_epoch_df.to_csv(mime_lr_epoch_csv_path, index=False)
    mime_lr_metrics_df.to_csv(mime_lr_metrics_csv_path, index=False)

    # mime test metrics
    mime_lr_final_test_metrics_data = []
    file_paths = glob.glob("MIME_logs/lr/*.log")
    for file_path in file_paths:
        mime_final_lr_test_metrics_data = extract_final_test_metrics(file_path, "lr")
        mime_lr_final_test_metrics_data.extend(mime_final_lr_test_metrics_data)
    mime_final_lr_test_metrics_df = pd.DataFrame(mime_lr_final_test_metrics_data)
    mime_final_lr_test_metrics_csv_path = "R_data/mime_lr_test_metrics.csv"
    mime_final_lr_test_metrics_df.to_csv(mime_final_lr_test_metrics_csv_path, index=False)


