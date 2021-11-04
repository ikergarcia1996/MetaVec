import subprocess
import os
from shlex import quote
import argparse
import logging
from typing import Dict, List


def get_environment(
    jiant_project_prefix: str, jiant_data_dir: str, word_embs_file: str
) -> Dict[str, str]:
    myenv: Dict[str, str] = os.environ.copy()
    myenv["JIANT_PROJECT_PREFIX"] = jiant_project_prefix
    myenv["JIANT_DATA_DIR"] = jiant_data_dir
    myenv["WORD_EMBS_FILE"] = word_embs_file
    return myenv


def remove_data(directory: str):
    for filename in os.listdir(directory):
        if not filename.endswith(".tsv"):
            if os.path.isfile(os.path.join(directory, filename)):
                os.remove(os.path.join(directory, filename))
            if os.path.isdir(os.path.join(directory, filename)):
                remove_data(os.path.join(directory, filename))


def results_2_csv(jiant_log_path: str, output_path: str):
    with open(jiant_log_path, "r", encoding="utf-8") as log_file, open(
        output_path, "w+", encoding="utf-8"
    ) as output_file:
        cola = (
            log_file.readline().rstrip().strip().split(",")[-2].split(":")[-1].strip()
        )
        ax = log_file.readline().rstrip().strip().split(",")[-2].split(":")[-1].strip()
        mnli = (
            log_file.readline().rstrip().strip().split(",")[-1].split(":")[-1].strip()
        )
        mrpc = log_file.readline().rstrip().strip().split(",")
        mrpc_acc = mrpc[3].split(":")[-1].strip()
        mrpc_f1 = mrpc[4].split(":")[-1].strip()
        qnli = (
            log_file.readline().rstrip().strip().split(",")[-1].split(":")[-1].strip()
        )
        qqp = log_file.readline().rstrip().strip().split(",")
        qqp_acc = qqp[3].split(":")[-1].strip()
        qqp_f1 = qqp[4].split(":")[-1].strip()
        rte = log_file.readline().rstrip().strip().split(",")[-1].split(":")[-1].strip()
        sst = log_file.readline().rstrip().strip().split(",")[-1].split(":")[-1].strip()
        sts = log_file.readline().rstrip().strip().split(",")
        sts_p = sts[-2].split(":")[-1].strip()
        sts_s = sts[-1].split(":")[-1].strip()
        wnli = (
            log_file.readline().rstrip().strip().split(",")[-1].split(":")[-1].strip()
        )

        results = [
            cola,
            sst,
            mrpc_f1,
            mrpc_acc,
            sts_p,
            sts_s,
            qqp_f1,
            qqp_acc,
            mnli,
            qnli,
            rte,
            wnli,
            ax,
        ]
        print(
            "cola-correlation,"
            "sst2-accuracy,"
            "mrpc-f1,mrpc-accuracy,"
            "stsb-pearson,stsb-spearman,"
            "qqp-f1,qqp-accuracy,"
            "mnli-accuracy,"
            "qnli-accuracy,"
            "rte-accuracy,"
            "wnli-accuracy,"
            "AX-correlation",
            file=output_file,
        )
        print(",".join(results), file=output_file)


def run_experiments(
    jiant_project_prefix: str,
    jiant_data_dir: str,
    word_embs_file: str,
    config_file: str,
    stats_path: str,
):

    logging.warning(
        "Remember to set allow_reuse_of_pretraining_parameters = 1 in jiant/jiant/config/defaults.conf"
    )

    print(
        f"jiant_project_prefix: {jiant_project_prefix}\n"
        f"jiant_data_dir: {jiant_data_dir}\n"
        f"word_embs_file: {word_embs_file}\n"
        f"config_file: {config_file}\n"
        f"stats_path: {stats_path}"
    )

    env = get_environment(
        jiant_project_prefix=jiant_project_prefix,
        jiant_data_dir=jiant_data_dir,
        word_embs_file=word_embs_file,
    )

    print(f"Environment: {env}")

    exp_name = (
        "".join(os.path.basename(word_embs_file).split(".")[:-1])
        + "_"
        + "".join(os.path.basename(config_file).split(".")[:-1])
    )
    command = (
        f"python3 {os.path.join(quote(jiant_project_prefix),'main.py')} "
        f"--config_file {quote(config_file)} "
        f'--overrides "exp_name = {quote(exp_name)} "'
    )
    print(command)

    jiant_exp_dir = os.path.join(jiant_project_prefix, exp_name)

    try:
        subprocess.check_call(["bash", "-c", command], env=env)
    except subprocess.CalledProcessError as err:
        logging.warning(f"Error running jiant. Exception:\n{err}")
        remove_data(jiant_exp_dir)
        return

    remove_data(jiant_exp_dir)

    if stats_path:
        if not os.path.exists(os.path.dirname(stats_path)):
            os.makedirs(os.path.dirname(stats_path))

    print(
        f"Exporting results to: { os.path.join(stats_path, 'results.csv') if not stats_path else stats_path}"
    )
    results_2_csv(
        os.path.join(jiant_exp_dir, "results.tsv"),
        os.path.join(stats_path, "results.csv") if not stats_path else stats_path,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jiant_project_prefix",
        type=str,
        default="jiant-v1-legacy",
        help="jiant_project_prefix",
    )

    parser.add_argument(
        "--jiant_data_dir",
        type=str,
        default="jiant-v1-legacy/data/",
        help="jiant_project_prefix",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i",
        "--word_embs_file",
        type=str,
        help="jiant_project_prefix",
    )

    group.add_argument(
        "-d",
        "--directory_path",
        type=str,
        help="Embedding to evaluate",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="jiant-v1-legacy/jiant/config/superglue_bow.conf",
        help="jiant_project_prefix",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="extrinsic_eval_results/",
        help="copy the final stats file to this path",
    )

    args = parser.parse_args()

    if args.word_embs_file:
        print(f"Evaluating: {args.word_embs_file}")

        stats_path = os.path.join(
            args.output_dir, os.path.basename(args.word_embs_file)
        )

        run_experiments(
            jiant_project_prefix=args.jiant_project_prefix,
            jiant_data_dir=args.jiant_data_dir,
            word_embs_file=args.word_embs_file,
            config_file=args.config_file,
            stats_path=stats_path,
        )

    else:
        emb_list: List[str] = [
            os.path.join(args.directory_path, f)
            for f in os.listdir(args.directory_path)
            if os.path.isfile(os.path.join(args.directory_path, f))
        ]

        print(f"Found {len(emb_list)} embeddings in directory: {emb_list}")

        for word_embs_file in emb_list:
            print(f"Evaluating embedding: {word_embs_file}")

            stats_path = os.path.join(args.output_dir, os.path.basename(word_embs_file))

            print(f"csv with results will be saved in: {stats_path}")

            run_experiments(
                jiant_project_prefix=args.jiant_project_prefix,
                jiant_data_dir=args.jiant_data_dir,
                word_embs_file=word_embs_file,
                config_file=args.config_file,
                stats_path=stats_path,
            )
