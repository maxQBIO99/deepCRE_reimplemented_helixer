import argparse
import os

import pandas as pd
from utils import predict

if not os.path.exists('results/predictions'):
    os.mkdir('results/predictions')

parser = argparse.ArgumentParser(
                    prog='deepCRE',
                    description="""
                    This script performs the deepCRE prediction. We assume you have the following three directories:
                    tmp_counts (contains your counts files), genome (contains the genome fasta files),
                    gene_models (contains the gtf files)
                    """)

parser.add_argument('--input',
                    help="""
                    This is a 5 column csv file with entries: genome, gtf, tpm, output name, number of chromosomes.""",
                    required=True)
parser.add_argument('--model_case', help="Can be SSC or SSR", required=True)
parser.add_argument('--ignore_small_genes', help="Ignore small genes, can be yes or no", required=True)

args = parser.parse_args()
data = pd.read_csv(args.input, sep=',', header=None,
                   dtype={0: str, 1: str, 2: str, 3: str, 4: str},
                   names=['genome', 'gtf', 'tpm', 'output', 'chroms'])
print(data.head())
if data.shape[1] != 5:
    raise Exception("Input file incorrect. Your input file must contain 5 columns and must be .csv")

for genome, gtf, tpm_counts, output_name, num_chromosomes in data.values:
    true_targets, preds, genes = [], [], []
    chromosomes = pd.read_csv(filepath_or_buffer=f'genome/{num_chromosomes}', header=None).values.ravel().tolist()
    for chrom in chromosomes:
        _, y, pred_probs, gene_ids, _ = predict(genome=genome, annot=gtf, tpm_targets=tpm_counts, upstream=1000,
                                                downstream=500, val_chromosome=str(chrom), output_name=output_name,
                                                model_case=args.model_case, ignore_small_genes=args.ignore_small_genes)
        true_targets.extend(y)
        preds.extend(pred_probs)
        genes.extend(gene_ids)

    result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
    print(result.head())
    result.to_csv(f'results/predictions/{output_name}_predictions.csv', index=False)
