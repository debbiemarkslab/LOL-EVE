import requests
import pandas as pd
import numpy as np
from io import StringIO
from Bio.Seq import Seq
from Bio import motifs
import gzip
import os
import sys
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import urllib.request
import urllib.error

NUM_CORES = 64

def fetch_tf_list(species="Homo sapiens", collection="CORE", filename="all_human_tfs.json"):
    if os.path.exists(filename):
        print(f"Loading TF list from existing file: {filename}")
        with open(filename, 'r') as f:
            return json.load(f)

    print("Fetching TF list from JASPAR...")
    base_url = "https://jaspar.genereg.net/api/v1/matrix/"
    params = {
        "species": species,
        "collection": collection,
        "format": "json"
    }
    
    all_results = []
    next_url = base_url

    while next_url:
        response = requests.get(next_url, params=params)
        response.raise_for_status()
        data = response.json()
        all_results.extend(data['results'])
        next_url = data['next']
        params = {}  # Clear params for subsequent requests

    tf_dict = {entry['name']: entry['matrix_id'] for entry in all_results}
    
    with open(filename, 'w') as f:
        json.dump(tf_dict, f, indent=2)
    
    return tf_dict

def get_gtex_data(filename="gtex_data.csv"):
    if os.path.exists(filename):
        print(f"Loading GTEx data from existing file: {filename}")
        gtex_data = pd.read_csv(filename, index_col=0)
    else:
        print("Fetching GTEx data...")
        gtex_file = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
        try:
            gtex_data = pd.read_csv(gtex_file, sep='\t', skiprows=2, compression='gzip')
            gtex_data = gtex_data.set_index('Description')
            gtex_data.to_csv(filename)
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print("Error: Unable to access GTEx data directly. This may be due to access restrictions.")
                print("Please follow these steps to obtain the GTEx data:")
                print("1. Visit https://gtexportal.org/home/datasets")
                print("2. Download the file 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gzt'")
                print("3. Place the downloaded file in the same directory as this script")
                print("4. Rename the file to 'gtex_data.csv.gz'")
                print("5. Run the script again")
                print("\nAlternatively, you can disable GTEx filtering by modifying the script.")
                sys.exit(1)
            else:
                raise

    # Check data types and values
    print("GTEx data shape:", gtex_data.shape)
    print("GTEx data types:\n", gtex_data.dtypes)
    print("Sample of GTEx data:\n", gtex_data.iloc[:5, :5])
    
    # Check for non-numeric values
    non_numeric = gtex_data.applymap(lambda x: not pd.api.types.is_numeric_dtype(type(x)))
    if non_numeric.any().any():
        print("Warning: Non-numeric values found in GTEx data")
        print(gtex_data[non_numeric].stack().value_counts())

    return gtex_data

def filter_tfs_by_expression(tf_dict, gtex_data, min_tissues=30, expression_threshold=1):
    filtered_tfs = {}
    for tf, jaspar_id in tf_dict.items():
        if tf in gtex_data.index:
            expression = gtex_data.loc[tf]
            if isinstance(expression, pd.Series):
                # Convert expression values to numeric, replacing non-numeric values with NaN
                expression = pd.to_numeric(expression, errors='coerce')
                # Count tissues where expression is above threshold, ignoring NaN values
                tissues_expressed = np.sum(expression > expression_threshold)
                if tissues_expressed >= min_tissues:
                    filtered_tfs[tf] = jaspar_id
            else:
                print(f"Warning: Unexpected data type for TF {tf}: {type(expression)}")
    return filtered_tfs
def write_tf_list_to_file(tf_dict, filename="selected_tfs.json"):
    with open(filename, 'w') as f:
        json.dump(tf_dict, f, indent=2)
    print(f"TF list written to {filename}")

def load_tf_list_from_file(filename="selected_tfs.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def fetch_and_process_motifs_to_pssms(tf_jaspar_ids):
    tf_motifs = {}
    tf_max_scores = {}

    for tf, jaspar_id in tf_jaspar_ids.items():
        url = f"https://jaspar.genereg.net/api/v1/matrix/{jaspar_id}/"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            pfm = {base: data['pfm'][base] for base in 'ACGT'}
            motif = motifs.Motif(alphabet='ACGT', counts=pfm)
            pssm = motif.counts.normalize(pseudocounts=0.5).log_odds()
            max_score = pssm.calculate(motif.consensus)
            tf_motifs[tf] = pssm
            tf_max_scores[tf] = max_score
        else:
            print(f"Failed to get motif for {tf}: {response.status_code}")

    return tf_motifs, tf_max_scores

def scan_sequence_for_tfbs(sequence, tf_motifs, tf_max_scores, threshold=0.8):
    tfbs_list = []
    sequence = Seq(sequence)

    for tf, pssm in tf_motifs.items():
        max_score = tf_max_scores[tf]
        threshold_score = threshold * max_score

        forward_scores = pssm.calculate(sequence)
        reverse_scores = pssm.calculate(sequence.reverse_complement())

        for i, (forward_score, reverse_score) in enumerate(zip(forward_scores, reverse_scores)):
            if forward_score >= threshold_score:
                tfbs_list.append((tf, i, i+len(pssm.consensus), forward_score, '+'))
            if reverse_score >= threshold_score:
                tfbs_list.append((tf, i, i+len(pssm.consensus), reverse_score, '-'))

    return tfbs_list

def find_tfbs_knockouts(sequence, tf_motifs, tf_max_scores):
    wt_tfbs = scan_sequence_for_tfbs(sequence, tf_motifs, tf_max_scores, 0.8)
    knockouts = []

    for tf, start, end, score, strand in wt_tfbs:
        deletion_length = end - start
        mutated_seq = sequence[:start] + sequence[end:]
        
        mutated_tfbs = scan_sequence_for_tfbs(mutated_seq, {tf: tf_motifs[tf]}, {tf: tf_max_scores[tf]}, 0.8)
        
        if not any(tfbs[0] == tf and tfbs[1] <= start and tfbs[2] >= end for tfbs in mutated_tfbs):
            knockouts.append((tf, start, end, strand))

    return knockouts

def process_promoter(promoter_data, tf_motifs, tf_max_scores):
    promoter_id, data = promoter_data
    sequence = data['sequence']
    chrom = data['chrom']
    start = data['start']

    knockouts = find_tfbs_knockouts(sequence, tf_motifs, tf_max_scores)

    results = []
    for tf, tf_start, tf_end, strand in knockouts:
        ref_seq = str(sequence[tf_start:tf_end])
        alt_seq = str(sequence[tf_start])
        info = f"TYPE=DEL;PROMOTER={promoter_id};EFFECT=KNOCKOUT;TF={tf};STRAND={strand}"
        vcf_record = f"{chrom}\t{start+tf_start}\t.\t{ref_seq}\t{alt_seq}\t.\tPASS\t{info}\n"
        results.append(vcf_record)

    return results

def write_vcf_batch(batch, task_id):
    output_dir = 'vcf_files/'
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f'tfbs_knockouts_task_{task_id}.vcf.gz')

    vcf_header = "##fileformat=VCFv4.2\n"
    vcf_header += "##INFO=<ID=TYPE,Number=1,Type=String,Description=\"Type of indel (DEL for TFBS knockouts)\">\n"
    vcf_header += "##INFO=<ID=PROMOTER,Number=1,Type=String,Description=\"ID of the promoter sequence\">\n"
    vcf_header += "##INFO=<ID=EFFECT,Number=1,Type=String,Description=\"Effect on TFBS (KNOCKOUT)\">\n"
    vcf_header += "##INFO=<ID=TF,Number=1,Type=String,Description=\"Transcription Factor affected\">\n"
    vcf_header += "##INFO=<ID=STRAND,Number=1,Type=String,Description=\"Strand of the TFBS (+ or -)\">\n"
    vcf_header += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"

    with gzip.open(filename, 'wt') as vcf_writer:
        vcf_writer.write(vcf_header)
        vcf_writer.writelines(batch)

def main(task_id):
    tf_file = f"selected_tfs_task_{task_id}.json"
    TF_JASPAR_IDS = load_tf_list_from_file(tf_file)

    if TF_JASPAR_IDS is None:
        print("Fetching TF list from JASPAR...")
        all_human_tfs = fetch_tf_list()
        print(f"Total human TFs fetched: {len(all_human_tfs)}")

        print("Fetching GTEx data...")
        gtex_data = get_gtex_data()

        print("GTEx data type:", type(gtex_data))
        print("GTEx data shape:", gtex_data.shape)
        print("GTEx index type:", type(gtex_data.index))
        print("Sample of GTEx index:", gtex_data.index[:5])
        print("Sample of GTEx data:\n", gtex_data.iloc[:5, :5])

        print("Filtering TFs based on expression...")
        TF_JASPAR_IDS = filter_tfs_by_expression(all_human_tfs, gtex_data)
        print(f"TFs after expression filtering: {len(TF_JASPAR_IDS)}")

        write_tf_list_to_file(TF_JASPAR_IDS, tf_file)
    else:
        print(f"Loaded {len(TF_JASPAR_IDS)} TFs from existing file: {tf_file}")

    print("Reading promoter sequences...")
    df = pd.read_csv('promoter_sequence.csv')

    promoter_sequences = {row.GENE: {'sequence': Seq(row.WT_SEQ), 'chrom': row.CHROM, 'start': int(row.START)}
                          for _, row in df.iterrows()}

    task_promoters = promoter_sequences

    print(f"Processing task {task_id} with {len(task_promoters)} promoters")

    print("Fetching and processing motifs...")
    tf_motifs, tf_max_scores = fetch_and_process_motifs_to_pssms(TF_JASPAR_IDS)

    all_results = []

    print("Processing promoters...")
    with ProcessPoolExecutor(NUM_CORES) as executor:
        jobs = []
        for promoter_data in task_promoters.items():
            job = executor.submit(process_promoter, promoter_data, tf_motifs, tf_max_scores)
            jobs.append(job)

        for job in tqdm(jobs, desc="Processing promoters"):
            results = job.result()
            all_results.extend(results)

    print("Writing VCF file...")
    write_vcf_batch(all_results, task_id)

    print(f"VCF file with TFBS knockouts has been generated for task {task_id}")

if __name__ == "__main__":
    main(0)
