import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import re, os, subprocess, tempfile
from collections import defaultdict
import argparse

STANDARD_GENETIC_CODE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K',
    'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W',
    'CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GGT':'G','GGC':'G','GGA':'G','GGG':'G'
}  # mantieni il tuo dizionario

# -------------------- FUNZIONI UTILITARIE --------------------

def clean_dna(seq):
    return "".join(c for c in seq.upper() if c in "ATGCN")

def get_longest_protein(orf_output):
    sequences, header, seq_list = {}, None, []
    for line in orf_output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header: sequences[header] = "".join(seq_list)
            header, seq_list = line, []
        else:
            seq_list.append(line)
    if header: sequences[header] = "".join(seq_list)
    if not sequences: return None, ""
    longest_header = max(sequences, key=lambda h: len(sequences[h]))
    return longest_header, sequences[longest_header]

def run_orffinder_on_sequence(seq_str):
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".fasta") as fin, \
         tempfile.NamedTemporaryFile(mode='r+', suffix=".gff") as fout:
        fin.write(">seq\n" + clean_dna(seq_str))
        fin.flush()
        cmd = ["./ORFfinder", "-in", fin.name, "-out", fout.name, "-outfmt", "0"]
        res = subprocess.run(cmd, check=True)
        fout.seek(0)
        return fout.read()

def load_genome(fasta_path):
    return {r.id: r.seq for r in SeqIO.parse(fasta_path, "fasta")}

def parse_gtf(gtf_path):
    gtf, gene_transcripts = {}, defaultdict(list)
    for line in open(gtf_path):
        if line.startswith("#"): continue
        parts = line.strip().split("\t")
        if len(parts) < 9: continue
        chrom, feat, start, end, strand, info = parts[0], parts[2], int(parts[3]), int(parts[4]), parts[6], parts[8]
        m_tx = re.search(r'transcript_id "([^"]+)"', info)
        if not m_tx: continue
        tx = m_tx.group(1)
        m_gene = re.search(r'gene_name "([^"]+)"', info) or re.search(r'gene_id "([^"]+)"', info)
        if not m_gene: continue
        gene = m_gene.group(1)
        if tx not in gtf: gtf[tx] = {"chrom": chrom, "strand": strand, "cds_exons": [], "exons": []}
        if feat == "CDS": gtf[tx]["cds_exons"].append((start,end))
        elif feat == "exon": gtf[tx]["exons"].append((start,end))
        gene_transcripts[gene].append(tx)
    # ordina exons/CDS per strand
    for tx, d in gtf.items():
        rev = d["strand"] == "-"
        d["cds_exons"].sort(key=lambda x:x[0], reverse=rev)
        d["exons"].sort(key=lambda x:x[0], reverse=rev)
    return gtf, dict(gene_transcripts)

def select_top_transcripts(gene_coords, gene_transcripts, max_transcripts=3):
    filtered_transcripts, filtered_gene_to_transcripts = {}, {}
    for gene, txs in gene_transcripts.items():
        ranking, seen = [], set()
        for tx in txs:
            if tx not in gene_coords: continue
            cds = tuple(sorted(gene_coords[tx]["cds_exons"]))
            if not cds or cds in seen: continue
            seen.add(cds)
            total_len = sum(e[1]-e[0]+1 for e in gene_coords[tx]["cds_exons"])
            ranking.append((tx, total_len, len(gene_coords[tx]["cds_exons"])))
        if not ranking: continue
        ranking.sort(key=lambda x:(x[1],x[2]), reverse=True)
        selected = [tx for tx,_,_ in ranking[:max_transcripts]] if str(max_transcripts).lower()!="all" else [tx for tx,_,_ in ranking]
        for tx in selected: filtered_transcripts[tx] = gene_coords[tx]
        filtered_gene_to_transcripts[gene] = selected
    return filtered_transcripts, filtered_gene_to_transcripts

def extract_cds_sequence(genome, transcript_entry, breakpoint, role, tol=0, mismatch='reject'):
    """
    Estrae sequenza CDS "head" o "tail" rispetto a un breakpoint, gestendo mismatch.
    
    Se il transcript non ha CDS, ritorna None.
    
    mismatch: 'reject', 'approximate', 'cut'
    """
    chrom = transcript_entry["chrom"]
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    if chrom not in genome:
        raise KeyError(f"Cromosoma {chrom} non trovato nel genome!")

    strand = transcript_entry["strand"]
    cds_exons = transcript_entry.get("cds_exons", [])
    if not cds_exons:
        # skip transcript senza CDS
        print(f"Attenzione: transcript {transcript_entry} non ha CDS, salto.")
        return Seq("")

    cds_exons = sorted(cds_exons, key=lambda x: x[0], reverse=(strand == "-"))
    exons = sorted(transcript_entry.get("exons", cds_exons), key=lambda x: x[0], reverse=(strand == "-"))
    print(exons[0],exons[-1])
    print(breakpoint)
    def is_on_junction(bp, exon_coords, tol=0):
        for i in range(len(exon_coords)-1):
            _, end = exon_coords[i]
            start_next, _ = exon_coords[i+1]
            if abs(bp - end) <= tol or abs(bp - start_next) <= tol:
                return True
        return False

    # controllo giunzioni e CDS
    on_junction = is_on_junction(breakpoint, exons, tol=tol)
    inside_cds = cds_exons[0][0] <= breakpoint <= cds_exons[-1][1]

    if not on_junction and not inside_cds:
        if mismatch == 'reject':
            return Seq("")
            raise ValueError(f"Breakpoint {breakpoint} non cade su giunzione esonica né dentro CDS!")
        elif mismatch == 'approximate':
            # approssima al CDS boundary più vicino
            distances = [abs(breakpoint - cds_exons[0][0]), abs(breakpoint - cds_exons[-1][1])]
            breakpoint = cds_exons[0][0] if distances[0] <= distances[1] else cds_exons[-1][1]
        elif mismatch == 'cut':
            pass  # taglia comunque la sequenza
        else:
            raise ValueError(f"Valore mismatch sconosciuto: {mismatch}")

    # costruisci sequenza CDS completa
    full_cds = Seq("")
    exon_positions = []  # (start,end,cDNA_start,cDNA_end)
    for start, end in cds_exons:
        piece = Seq(str(genome[chrom][start-1:end]))
        exon_start = len(full_cds)
        full_cds += piece
        exon_end = len(full_cds)
        exon_positions.append((start, end, exon_start, exon_end))

    if strand == "-":
        full_cds = full_cds.reverse_complement()

    # trova cut point
    cut = None
    if strand == "+":
        for start, end, c_start, c_end in exon_positions:
            if start <= breakpoint <= end:
                cut = c_start + (breakpoint - start)
                break
    else:
        for start, end, c_start, c_end in exon_positions:
            if start <= breakpoint <= end:
                cut = len(full_cds) - (c_start + (breakpoint - start))
                break

    if cut is None:
        raise ValueError(f"Breakpoint {breakpoint} non trovato sulle CDS del transcript!")

    cds_seq = full_cds[:cut] if role == "head" else full_cds[cut:]
    return cds_seq



def extract_cds_sequence2(genome, transcript_entry, breakpoint, role,
                         tol=5, mismatch='reject', intron=False):
    """
    Estrae sequenza CDS "head" o "tail" rispetto a un breakpoint, con stampe di debug.
    
    Parametri:
    - genome: dict(chr -> string sequenza)
    - transcript_entry: dict con chiavi 'chrom', 'strand', 'cds_exons', 'exons'
    - breakpoint: int, posizione genomica
    - role: "head" o "tail"
    - tol: tolleranza per giunzioni
    - mismatch: come gestire bp fuori da CDS ['reject', 'approximate', 'cut']
    - intron: bool, se True includo introni
    """
    chrom = transcript_entry["chrom"]
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    if chrom not in genome:
        print(f"Chromosome {chrom} not found")
        return Seq("")
        raise KeyError(f"Cromosoma {chrom} non trovato nel genome!")

    strand = transcript_entry["strand"]
    cds_exons = transcript_entry.get("cds_exons", [])
    exons = transcript_entry.get("exons", cds_exons)

    #print(f"Chromosome: {chrom}, Strand: {strand}")
    if cds_exons:
        #print(f"CDS exons: {cds_exons[0]} ... {cds_exons[-1]}")
        pass
    #print(f"Breakpoint: {breakpoint}")
    if not cds_exons:
        print(f"[DEBUG] Transcript senza CDS: {transcript_entry}")
        return Seq("")

    # ordino esoni
    cds_exons = sorted(cds_exons, key=lambda x: x[0], reverse=(strand == "-"))
    exons = sorted(exons, key=lambda x: x[0], reverse=(strand == "-"))
    #print(f"Exons: {exons[0]} ... {exons[-1]}")

    # check se bp è vicino a giunzione
    def is_on_junction(bp, exon_coords, tol=0):
        for i in range(len(exon_coords)-1):
            _, end = exon_coords[i]
            start_next, _ = exon_coords[i+1]
            if abs(bp - end) <= tol or abs(bp - start_next) <= tol:
                return True
        return False

    on_junction = is_on_junction(breakpoint, exons, tol=tol)
    inside_cds = any(start <= breakpoint <= end for start, end in cds_exons)

    #print(f"[DEBUG] On junction: {on_junction}, Inside CDS: {inside_cds}")

    # gestione mismatch / introni
    if not on_junction and not inside_cds:
        if mismatch == 'reject':
            print(f"[DEBUG] Breakpoint {breakpoint} non su CDS o giunzione → reject")
            return Seq("")
        elif mismatch == 'approximate':
            # proietto sul CDS più vicino
            closest_bp = min(
                (b for start, end in cds_exons for b in (start, end)),
                key=lambda x: abs(x - breakpoint)
            )
            #print(f"[DEBUG] Breakpoint proiettato da {breakpoint} a {closest_bp} (approximate)")
            breakpoint = closest_bp
        elif mismatch == 'cut':
            if not inside_cds and not intron:
                print(f"[DEBUG] Breakpoint {breakpoint} in introne, intron=False → salto")
                return Seq("")
        else:
            raise ValueError(f"Valore mismatch sconosciuto: {mismatch}")

    # costruisco sequenza CDS completa
    full_cds = Seq("")
    exon_positions = []  # (start, end, cDNA_start, cDNA_end)
    for start, end in cds_exons:
        piece = Seq(str(genome[chrom][start-1:end]))
        exon_start = len(full_cds)
        full_cds += piece
        exon_end = len(full_cds)
        exon_positions.append((start, end, exon_start, exon_end))

    if strand == "-":
        full_cds = full_cds.reverse_complement()

    # calcolo cut point
    cut = None
    closest_dist = float("inf")
    closest_cut = None

    for start, end, c_start, c_end in exon_positions:
        if start <= breakpoint <= end:
            # breakpoint dentro esone
            cut = c_start + (breakpoint - start) if strand == "+" else len(full_cds) - (c_start + (breakpoint - start))
            break
        # controllo distanza più vicina
        for g_pos, cds_pos in [(start, c_start), (end, c_start + (end - start))]:
            dist = abs(breakpoint - g_pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_cut = cds_pos

    if cut is None:
        cut = closest_cut
        #print(f"[DEBUG] Cut point non esatto, uso closest_cut = {cut}")

    cds_seq = full_cds[:cut] if role == "head" else full_cds[cut:]
    print(f"[DEBUG] CDS {role} length: {len(cds_seq)}")
    return cds_seq

def read_file(infile):
    ext = os.path.splitext(infile)[1].lower()
    if ext in [".csv"]: return pd.read_csv(infile)
    if ext in [".tsv",".txt"]: return pd.read_csv(infile, sep="\t")
    if ext in [".pkl",".pickle"]: return pd.read_pickle(infile)
    if ext in [".xlsx",".xls"]: return pd.read_excel(infile)
    if ext in [".json"]: return pd.read_json(infile)
    if ext in [".jsonl",".ndjson"]: return pd.read_json(infile, lines=True)
    if ext in [".parquet"]: return pd.read_parquet(infile)
    if ext in [".feather"]: return pd.read_feather(infile)
    if ext in [".h5",".hdf5"]: return pd.read_hdf(infile)
    raise ValueError(f"Unsupported file type {ext}")

def write_file(df,outfile):
    df.drop_duplicates(subset="sequence",inplace=True)
    ext = os.path.splitext(outfile)[1].lower()
    if ext in [".csv"]: df.to_csv(outfile,index=False)
    elif ext in [".tsv",".txt"]: df.to_csv(outfile, sep="\t", index=False)
    elif ext in [".pkl",".pickle"]: df.to_pickle(outfile)
    elif ext in [".xlsx",".xls"]: df.to_excel(outfile,index=False)
    elif ext in [".json"]: df.to_json(outfile,orient="records")
    elif ext in [".jsonl",".ndjson"]: df.to_json(outfile, orient="records", lines=True)
    elif ext in [".parquet"]: df.to_parquet(outfile,index=False)
    elif ext in [".feather"]: df.reset_index(drop=True).to_feather(outfile)
    elif ext in [".h5",".hdf5"]: df.to_hdf(outfile, key="df", mode="w")
    else: raise ValueError(f"Unsupported output {ext}")
    print(f"File {outfile} saved succesfully, contains {len(df)} lines")

# -------------------- MAIN --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", required=True)
    parser.add_argument("--gtf_path", required=True)
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--junction", choices=["reject","cut","approximate"], default="reject")
    parser.add_argument("--tol", type=int, default=5)
    parser.add_argument("--intron", action="store_true")
    parser.add_argument("--no-translate", action="store_true")
    parser.add_argument("--transcript_columns", type=str)
    parser.add_argument("--gene_columns", type=str)
    parser.add_argument("--breakpoint_columns", type=str, required=True)
    args = parser.parse_args()

    genome = load_genome(args.fasta_path)
    gtf_raw, gene_transcripts_raw = parse_gtf(args.gtf_path)
    gtf, gene_transcripts = select_top_transcripts(gtf_raw, gene_transcripts_raw, max_transcripts=3)
    translate = not args.no_translate
    df = read_file(args.infile)
    print(f"Processing input with {len(df)} records")
    print(df.head())
    
    up_gene_col,dw_gene_col = args.gene_columns.split(",")
    up_trans_col,dw_trans_col = args.transcript_columns.split(",") if args.transcript_columns else (None,None)
    up_bp_col,dw_bp_col = args.breakpoint_columns.split(",")

    def get_tx_list(gene_val, tx_val):
        if pd.notnull(tx_val) and tx_val != '.':
            return [tx_val]
        txs = []
        for g in str(gene_val).split(","):
            txs += gene_transcripts.get(g.strip(), [])
        return txs

    temp_file = os.path.splitext(args.outfile)[0]+"_temp.tsv"
    with open(temp_file,"w") as f:
        header = list(df.columns)
        if not args.transcript_columns: header += ["trans_h","trans_t"]
        header += ["sequence","sequence_length"]
        f.write("\t".join(header)+"\n")

        for idx,row in df.iterrows():
            if idx%50==0: print(f"Processing line {idx}/{len(df)}")
            tx_heads = get_tx_list(row[up_gene_col], row[up_trans_col])
            tx_tails = get_tx_list(row[dw_gene_col], row[dw_trans_col])
            #print(f"Processing record {idx} with {row[up_trans_col]} and {row[dw_trans_col]}")
            #print(f"Considering transcripts {tx_heads} and {tx_tails}")

            for tx_h in tx_heads:
                for tx_t in tx_tails:
                    try:
                        t_h=gtf_raw[tx_h]
                        t_t=gtf_raw[tx_t]
                    except: continue
                    dna_h = extract_cds_sequence2(genome, t_h, int(row[up_bp_col]), "head",
                                                    tol=args.tol, mismatch=args.junction)
                    print(len(dna_h))
                    dna_t = extract_cds_sequence2(genome, t_t, int(row[dw_bp_col]), "tail",
                                                    tol=args.tol, mismatch=args.junction)
                    print(len(dna_t))
                    fusion_dna = clean_dna(str(dna_h+dna_t))
                    seq = fusion_dna
                    if translate:
                        _, seq = get_longest_protein(run_orffinder_on_sequence(fusion_dna))
                        if not seq: continue
                    row_vals = [str(row[c]) for c in df.columns]
                    if not args.transcript_columns: row_vals += [tx_h,tx_t]
                    row_vals += [seq,str(len(seq))]
                    f.write("\t".join(row_vals)+"\n")

    write_file(pd.read_csv(temp_file, sep="\t"), args.outfile)


"""
python extract_cpu.py \
    --fasta_path /homes/vmelotti/project/data/genome/hg19.fa \
    --gtf_path /homes/vmelotti/project/data/genome/Homo_sapiens.GRCh37.87.gtf \
    --infile /homes/vmelotti/project/data/raw/data_SRR_cancer.pkl \
    --outfile /homes/vmelotti/project/data/raw/data_SRR_cancer_sequence.pkl \
    --transcript_columns transcript_id1,transcript_id2 \
    --breakpoint_columns bp1,bp2 \
    --gene_columns gene1,gene2 \
    --junction approximate


    
python extract_cpu.py \
    --fasta_path /homes/vmelotti/project/data/genome/hg19.fa \
    --gtf_path /homes/vmelotti/project/data/genome/Homo_sapiens.GRCh37.87.gtf \
    --infile /homes/vmelotti/project/data/raw/data_decider_positive_WITH_SEQUENCE.pkl \
    --outfile /homes/vmelotti/project/data/raw/data_decider_positive_WITH_SEQUENCE.pkl \
    --transcript_columns transcript_id1,transcript_id2 \
    --breakpoint_columns Breakpoint1,Breakpoint2 \
    --gene_columns Gene1,Gene2 \
    --junction approximate
"""
