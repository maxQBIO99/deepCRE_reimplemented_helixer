import argparse
import pandas as pd
from utils import generate_motifs
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
tf.config.set_visible_devices([], 'GPU')

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
    generate_motifs(genome=genome, annot=gtf, tpm_targets=tpm_counts, upstream=1000, downstream=500,
                    ignore_small_genes=args.ignore_small_genes, output_name=output_name,
                    model_case=args.model_case, n_chromosomes=num_chromosomes)
