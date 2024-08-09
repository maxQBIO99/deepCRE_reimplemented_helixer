import argparse
import os
import pandas as pd
from utils import train_deep_cre
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

parser = argparse.ArgumentParser(
                    prog='deepCRE',
                    description="""
                    This script performs the deepCRE training. We assume you have the following three directories:
                    tmp_counts (contains your counts files), genome (contains the genome fasta files),
                    gene_models (contains the gtf files)
                    """)
parser.add_argument('--input',
                    help="""
                    This is a six column csv file with entries: genome, gtf, tpm, output name,
                    number of chromosomes and pickle_key.""", required=True)
parser.add_argument('--pickle', help="path to pickle file", required=True)
parser.add_argument('--model_case', help="Can be SSC or SSR", required=True)
parser.add_argument('--ignore_small_genes', help="Ignore small genes, can be yes or no", required=True)

args = parser.parse_args()

data = pd.read_csv(args.input, sep=',', header=None,
                   dtype={0: str, 1: str, 2: str, 3: str, 4: int, 5: str},
                   names=['genome', 'gtf', 'tpm', 'output', 'counts', 'p_key'])
print(data.head())
if data.shape[1] != 6:
    raise Exception("Input file incorrect. Your input file must contain 6 columns and must be .csv")


for genome, gtf, tpm_counts, output_name, num_chromosomes, pickled_key in data.values:
    results_genome = []
    for val_chrom in range(1, num_chromosomes + 1):
        results = train_deep_cre(genome=genome,
                                 annot=gtf,
                                 tpm_targets=tpm_counts,
                                 upstream=1000,
                                 downstream=500,
                                 genes_picked=args.pickle,
                                 val_chromosome=str(val_chrom),
                                 output_name=output_name,
                                 model_case=args.model_case,
                                 pickled_key=pickled_key,
                                 ignore_small_genes=args.ignore_small_genes)
        results_genome.append(results)
        print(f"Results for genome: {genome}, chromosome: {val_chrom}: {results}")
    results_genome = pd.DataFrame(results_genome, columns=['loss', 'accuracy', 'auROC', 'auPR'])
    results_genome.to_csv(path_or_buf=f'results/{args.model_case}_{output_name}_results.csv', index=False)
    print(results_genome.head())








