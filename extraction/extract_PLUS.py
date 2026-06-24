import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import re
from collections import defaultdict
import subprocess
import tempfile
import argparse
import os

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
}

def get_longest_protein(orf_output):

    sequences = {}
    current_header = None
    current_seq = []

    for line in orf_output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                sequences[current_header] = "".join(current_seq)
            current_header = line
            current_seq = []
        else:
            current_seq.append(line)
    
    if current_header is not None:
        sequences[current_header] = "".join(current_seq)

    if not sequences:
        return None, ""
    
    longest_header = max(sequences, key=lambda h: len(sequences[h]))
    return longest_header, str(sequences[longest_header])

def run_orffinder_on_sequence(sequence_str):
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".fasta") as temp_input, \
         tempfile.NamedTemporaryFile(mode='r+', suffix=".gff") as temp_output:

        temp_input.write(">seq\n")
        temp_input.write(clean_dna(sequence_str))
        temp_input.flush()  

        cmd = ["./ORFfinder", "-in", temp_input.name,
               "-out", temp_output.name, "-outfmt", "0"]

        result=subprocess.run(cmd, check=True)
        
        if result.returncode != 0:
            print(result.stderr)
            print("FASTA file:", temp_input.name)
            raise RuntimeError("ORFfinder failed")

        temp_output.seek(0)
        return temp_output.read()

def parse_gtf(gtf_path):
    gene_coords = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            chrom, feature, start, end, strand, info = parts[0], parts[2], int(parts[3]), int(parts[4]), parts[6], parts[8]

            match = re.search(r'transcript_id "([^"]+)"', info)
            if not match:
                continue
            tx_id = match.group(1)

            if tx_id not in gene_coords:
                gene_coords[tx_id] = {"chrom": chrom, "strand": strand, "cds_exons": [], "exons": []}

            if feature == "CDS":
                gene_coords[tx_id]["cds_exons"].append((start, end))
            elif feature == "exon":
                gene_coords[tx_id]["exons"].append((start, end))

    # ordina gli esoni e CDS per strand
    for tx_id, data in gene_coords.items():
        if data["strand"] == "+":
            data["cds_exons"].sort(key=lambda x: x[0])
            data["exons"].sort(key=lambda x: x[0])
        else:
            data["cds_exons"].sort(key=lambda x: x[0], reverse=True)
            data["exons"].sort(key=lambda x: x[0], reverse=True)

    return gene_coords

def extract_cds_sequence(genome, transcript_entry, breakpoint, role,
                         tol=5, mismatch='reject', intron=False):
    """
    Estrae sequenza CDS "head" o "tail" rispetto a un breakpoint, gestendo mismatch e introni.
    
    Parametri:
    - genome: dizionario {chrom: sequence}
    - transcript_entry: dict con chiavi 'chrom', 'strand', 'cds_exons', 'exons'
    - breakpoint: coordinata genomica
    - role: "head" o "tail"
    - tol: tolleranza per considerare giunzioni
    - mismatch: 'reject', 'approximate', 'cut'
    - intron: True → taglia anche in introni, False → scarta fusioni in introni

    Ritorna:
    - Seq oggetto della sequenza estratta (vuoto se scartata)
    """

    chrom = transcript_entry["chrom"]
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    if chrom not in genome:
        raise KeyError(f"Cromosoma {chrom} non trovato nel genome!")

    strand = transcript_entry["strand"]
    cds_exons = transcript_entry.get("cds_exons", [])
    if not cds_exons:
        print(f"Attenzione: transcript {transcript_entry} non ha CDS, salto.")
        return Seq("")

    # ordino esoni
    cds_exons = sorted(cds_exons, key=lambda x: x[0], reverse=(strand == "-"))
    exons = sorted(transcript_entry.get("exons", cds_exons),
                   key=lambda x: x[0], reverse=(strand == "-"))

    # funzione di controllo giunzioni
    def is_on_junction(bp, exon_coords, tol=0):
        for i in range(len(exon_coords)-1):
            _, end = exon_coords[i]
            start_next, _ = exon_coords[i+1]
            if abs(bp - end) <= tol or abs(bp - start_next) <= tol:
                return True
        return False

    on_junction = is_on_junction(breakpoint, exons, tol=tol)
    inside_cds = any(start <= breakpoint <= end for start, end in cds_exons)

    # gestione mismatch/intron
    if not on_junction and not inside_cds:
        if mismatch == 'reject':
            raise ValueError(f"Breakpoint {breakpoint} non cade su giunzione esonica né dentro CDS!")
        elif mismatch == 'approximate':
            # proietta sul CDS boundary più vicino
            closest_bp = min(
                (b for start, end in cds_exons for b in (start, end)),
                key=lambda x: abs(x - breakpoint)
            )
            breakpoint = closest_bp
        elif mismatch == 'cut':
            if not inside_cds and not intron:
                print(f"Breakpoint {breakpoint} cade in introne e intron=False → salto")
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

    # trova cut point
    cut = None
    closest_dist = float("inf")
    closest_cut = None

    if strand == "+":
        for start, end, c_start, c_end in exon_positions:
            if start <= breakpoint <= end:
                cut = c_start + (breakpoint - start)
                break
            for g_pos, cds_pos in [(start, c_start), (end, c_start + (end - start))]:
                dist = abs(breakpoint - g_pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_cut = cds_pos
    else:  # strand "-"
        for start, end, c_start, c_end in exon_positions:
            if start <= breakpoint <= end:
                cut = len(full_cds) - (c_start + (breakpoint - start))
                break
            for g_pos, cds_pos in [(start, len(full_cds) - c_start),
                                   (end, len(full_cds) - (c_start + (end - start)))]:
                dist = abs(breakpoint - g_pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_cut = cds_pos

    if cut is None:
        cut = closest_cut

    cds_seq = full_cds[:cut] if role == "head" else full_cds[cut:]
    return cds_seq

def clean_dna(seq):
    return "".join([c for c in seq.upper() if c in "ATGCN"])

def is_on_exon_junction(breakpoint, exon_coords, tol=0):
    """
    breakpoint: coordinata genomica
    exon_coords: lista di (start,end) per tutti gli esoni del trascritto
    tol: tolleranza in bp (es. 2 basi)
    """
    for i in range(len(exon_coords)-1):
        _, end = exon_coords[i]
        start_next, _ = exon_coords[i+1]
        if abs(breakpoint - end) <= tol or abs(breakpoint - start_next) <= tol:
            return True
    return False

def load_genome(fasta_path):
    genome = {}
    with open(fasta_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            genome[record.id] = record.seq
    return genome

def reverse_complement(seq):
    complement = str.maketrans('ATCGatcg','TAGCtagc')
    return seq.translate(complement)[::-1]

def translate_seq(dna_seq, genetic_code=STANDARD_GENETIC_CODE):
    protein = ""
    for i in range(0,len(dna_seq)-2,3):
        codon = dna_seq[i:i+3].upper()
        protein += genetic_code.get(codon,'X')
    return protein

def parse_gtf_basic_protein_coding(gtf_path):
    """
    Parsing GTF che:
      - considera solo transcript con tag "basic" E transcript_biotype "protein_coding"
      - costruisce gene_coords (per transcript_id) con chrom, strand, cds_exons, exons
      - costruisce gene_to_transcripts: mapping gene_name (e gene_synonym) -> list of transcript_id
      - assicura che gene_name venga letto anche dalle righe 'transcript'
    Ritorna: gene_coords, gene_to_transcripts
    """
    # regex precompilate
    rx_tx = re.compile(r'transcript_id "([^"]+)"')
    rx_gene = re.compile(r'gene_name "([^"]+)"')
    rx_syn = re.compile(r'gene_synonym "([^"]+)"')
    rx_tag = re.compile(r'tag "([^"]+)"')
    rx_biotype = re.compile(r'transcript_biotype "([^"]+)"')

    # 1) prima passata: raccogli tutti i transcript basic ∧ protein_coding
    basic_pc_txs = set()
    # Inoltre salviamo i gene_name e synonyms che troviamo sulle righe transcript
    tx_to_gene = {}       # tx_id -> gene_name (se presente)
    tx_to_synonyms = {}   # tx_id -> [syn1, syn2...]

    with open(gtf_path, 'r') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature != "transcript":
                continue
            attrs = parts[8]

            m_tx = rx_tx.search(attrs)
            if not m_tx:
                continue
            tx = m_tx.group(1)

            # biotype
            m_bio = rx_biotype.search(attrs)
            if not m_bio:
                continue
            biotype = m_bio.group(1)
            if biotype != "protein_coding":
                continue

            # tags: possono essere multipli, campo 'tag' può essere ripetuto
            tags = [t.strip() for t in rx_tag.findall(attrs)]
            # rx_tag.findall restituirà elementi come 'CCDS' e 'basic' separatamente,
            # ma in alcuni GTF il tag è 'tag "CCDS","basic"' o più tag; coveriamo entrambe le casistiche
            expanded_tags = []
            for t in tags:
                for sub in t.split(','):
                    expanded_tags.append(sub.strip().strip('"').lower())

            if "basic" not in expanded_tags:
                continue

            # ok: è basic e protein_coding
            basic_pc_txs.add(tx)

            # salva gene_name/syn se presenti sulla riga transcript (utile per mappature)
            m_gene = rx_gene.search(attrs)
            if m_gene:
                tx_to_gene[tx] = m_gene.group(1)
            m_syn = rx_syn.search(attrs)
            if m_syn:
                tx_to_synonyms[tx] = [s.strip() for s in m_syn.group(1).split(',') if s.strip()]

    # 2) seconda passata: costruisci gene_coords SOLO per questi transcript,
    # ma usa gene_name anche dalle righe 'transcript' quando presenti
    gene_coords = {}
    gene_to_transcripts = defaultdict(list)

    with open(gtf_path, 'r') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9:
                continue

            chrom = parts[0]
            feature = parts[2]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attrs = parts[8]

            m_tx = rx_tx.search(attrs)
            if not m_tx:
                continue
            tx = m_tx.group(1)

            if tx not in basic_pc_txs:
                continue  # solo i transcript filtrati

            # Inizializza struttura se prima volta
            if tx not in gene_coords:
                gene_coords[tx] = {
                    "chrom": chrom,
                    "strand": strand,
                    "cds_exons": [],
                    "exons": []
                }

            if feature == "CDS":
                gene_coords[tx]["cds_exons"].append((start, end))
            elif feature == "exon":
                gene_coords[tx]["exons"].append((start, end))

            # Determina gene_name: prima prova a leggere da tx_to_gene salvato prima,
            # altrimenti prendi gene_name dalla riga corrente se presente.
            gene_name = tx_to_gene.get(tx)
            if gene_name is None:
                m_gene = rx_gene.search(attrs)
                if m_gene:
                    gene_name = m_gene.group(1)

            # synonyms: prima da tx_to_synonyms, altrimenti dalla riga corrente
            synonyms = tx_to_synonyms.get(tx, [])
            if not synonyms:
                m_syn = rx_syn.search(attrs)
                if m_syn:
                    synonyms = [s.strip() for s in m_syn.group(1).split(',') if s.strip()]

            # mapping gene -> transcript (usando gene_name e synonyms)
            if gene_name:
                gene_to_transcripts[gene_name].append(tx)
            for s in synonyms:
                gene_to_transcripts[s].append(tx)

    # ordina esoni e CDS rispettando strand
    # rimuove eventuali transcript che non hanno CDS raccolti (se vuoi mantenerli commenta la riga che fa del gene_coords del)
    for tx, data in list(gene_coords.items()):
        if len(data["cds_exons"]) == 0:
            # se non trovi CDS, probabilmente transcript non codificante; rimuovi
            del gene_coords[tx]
            # rimuovi tx anche dalle mappe (se presenti)
            for g in list(gene_to_transcripts.keys()):
                gene_to_transcripts[g] = [t for t in gene_to_transcripts[g] if t != tx]
                if not gene_to_transcripts[g]:
                    del gene_to_transcripts[g]
            continue

        if data["strand"] == "+":
            data["cds_exons"].sort(key=lambda x: x[0])
            data["exons"].sort(key=lambda x: x[0])
        else:
            data["cds_exons"].sort(key=lambda x: x[0], reverse=True)
            data["exons"].sort(key=lambda x: x[0], reverse=True)

    return gene_coords, dict(gene_to_transcripts)

def write_file(df, outfile):

    ext = os.path.splitext(outfile)[-1].lower()

    try:
        # CSV standard
        if ext in [".csv"]:
            df.to_csv(outfile, index=False)
        
        # TSV o TXT
        elif ext in [".tsv", ".txt"]:
            df.to_csv(outfile, sep="\t", index=False)
        
        # Pickle
        elif ext in [".pkl", ".pickle"]:
            df.to_pickle(outfile)
        
        # Excel
        elif ext in [".xlsx", ".xls"]:
            df.to_excel(outfile, index=False)
        
        # JSON
        elif ext in [".json"]:
            df.to_json(outfile, orient="records", lines=False)
        
        # JSON line (streaming)
        elif ext in [".jsonl", ".ndjson"]:
            df.to_json(outfile, orient="records", lines=True)
        
        # Parquet
        elif ext in [".parquet"]:
            df.to_parquet(outfile, index=False)
        
        # Feather
        elif ext in [".feather"]:
            df.reset_index(drop=True, inplace=True)  # necessario per Feather
            df.to_feather(outfile)
        
        # HDF5
        elif ext in [".h5", ".hdf5"]:
            df.to_hdf(outfile, key="df", mode="w")
        
        # Compressed CSV supportato da pandas
        elif ext in [".gz", ".bz2"]:
            df.to_csv(outfile, index=False, compression=ext[1:])  # ext senza punto
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        raise RuntimeError(f"Error writing file {outfile}: {e}")
    print(f"File {outfile} was saved successfully, with {len(df)} lines.")

def read_file(infile):
    ext = os.path.splitext(infile)[-1].lower()

    try:
        if ext in [".csv"]:
            df = pd.read_csv(infile)

        elif ext in [".tsv", ".txt"]:
            df = pd.read_csv(infile, sep="\t")

        elif ext in [".pkl", ".pickle"]:
            df = pd.read_pickle(infile)

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(infile)

        elif ext in [".json"]:
            df = pd.read_json(infile)

        elif ext in [".jsonl", ".ndjson"]:
            df = pd.read_json(infile, lines=True)

        elif ext in [".parquet"]:
            df = pd.read_parquet(infile)

        elif ext in [".feather"]:
            df = pd.read_feather(infile)

        elif ext in [".h5", ".hdf5"]:
            df = pd.read_hdf(infile)

        elif ext in [".gz", ".bz2"]:
            df = pd.read_csv(infile)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {infile}: {e}")
    
    return df

def select_top_transcripts(gene_coords, gene_to_transcripts, max_transcripts=3):
    """
    Seleziona al massimo N transcript per ciascun gene,
    rimuovendo transcript con le stesse identiche coordinate CDS,
    ordinati per:
        1. lunghezza CDS totale
        2. numero di CDS exons
        3. posizione genomica (euristica tie-break)
    
    Se max_transcripts='all', prende tutti i transcript unici.
    
    Ritorna:
        - filtered_transcripts: dizionario dove la chiave è il `transcript_id`
          e il valore è il dato associato (coordinate, exons, ecc.)
        - filtered_gene_to_transcripts: dizionario che mappa `gene_name` a
          una lista dei `transcript_id` selezionati.
    """
    filtered_transcripts = {}  
    filtered_gene_to_transcripts = {}  

    for gene, tx_list in gene_to_transcripts.items():
        ranking = []
        seen_cds_sets = set()  

        for tx in tx_list:
            if tx not in gene_coords:
                continue

            cds_exons = gene_coords[tx]["cds_exons"]
            if not cds_exons:
                continue

            cds_tuple = tuple(sorted(cds_exons))
            if cds_tuple in seen_cds_sets:
                continue  
            seen_cds_sets.add(cds_tuple)

            total_len = sum(e[1] - e[0] + 1 for e in cds_exons)
            exon_count = len(cds_exons)
            leftmost = min(e[0] for e in cds_exons)  

            ranking.append((tx, total_len, exon_count, leftmost))

        if not ranking:
            continue

        ranking.sort(key=lambda x: (x[1], x[2], -x[3]), reverse=True)

        if str(max_transcripts).strip().lower() == 'all':
            selected_txs = [tx for tx, _, _, _ in ranking]
        else:
            selected_txs = [tx for tx, _, _, _ in ranking[:int(max_transcripts)]]

        for tx in selected_txs:
            filtered_transcripts[tx] = gene_coords[tx]

        filtered_gene_to_transcripts[gene] = selected_txs

    return filtered_transcripts, filtered_gene_to_transcripts

def parse_args():
    parser = argparse.ArgumentParser(description="Extract CDS sequences and handle breakpoints for fusion genes.")

    parser.add_argument("--fasta_path", type=str, required=True,
                        help="Path to the input genome FASTA file.")
    parser.add_argument("--gtf_path", type=str, required=True,
                        help="Path to the input GTF annotation file.")
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to the input table with fusion or transcript data.")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Path where the output CDS / AA sequences will be written.")
    parser.add_argument("--junction", choices=["reject", "cut", "approximate"],
                        default="reject",
                        help="How to handle breakpoints relative to exon junctions: reject, cut anyway, or approximate to nearest junction.")
    parser.add_argument("--tol", type=int, default=5,
                        help="Tolerance in base pairs for considering exon junctions.")
    parser.add_argument("--intron", action="store_true",
                        help="Include intronic regions")
    parser.add_argument("--no-translate", action="store_true",
                        help="Disable translation to protein")
    parser.add_argument("--transcript_columns",type=str,
                        help='Transcript column names, if present, separated by comma')
    parser.add_argument("--gene_columns",type=str,required=True,
                        help='Gene column names, separated by comma')
    parser.add_argument("--breakpoint_columns",type=str,required=True,
                        help='Breakpoint column names, separated by comma')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    print("Loading genome and GTF...")
    genome = load_genome(args.fasta_path)

    # parse GTF e selezione top transcripts
    if args.transcript_columns is not None:
        gtf, gene_transcripts = parse_gtf(args.gtf_path)
    else:
        gtf, gene_transcripts = parse_gtf_basic_protein_coding(args.gtf_path)
        gtf, gene_transcripts = select_top_transcripts(gtf, gene_transcripts, max_transcripts=3)

    translate = not args.no_translate
    print("Done!")
    df = read_file(args.infile)

    # colonne input
    up_gene_col, dw_gene_col = args.gene_columns.split(",")

    if args.transcript_columns is not None:
        up_trans_col, dw_trans_col = args.transcript_columns.split(",")
    else:
        df["trans_h"]=None
        df["trans_t"]=None
        up_trans_col,dw_trans_col="trans_h","trans_t"
    up_breakpoint_col, dw_breakpoint_col = args.breakpoint_columns.split(",")

    temp_file = os.path.splitext(args.outfile)[0] + "_temp.tsv"
    print("Starting extraction...")
    with open(temp_file, "w") as f:

        # header
        header = list(df.columns)
        if args.transcript_columns is None:
            header += ["trans_h", "trans_t"]
        header += ["sequence", "sequence_length"]
        f.write("\t".join(header) + "\n")

        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"Processing line {idx}/{len(df)}")

            bp_head = row[up_breakpoint_col]
            bp_tail = row[dw_breakpoint_col]

            # costruzione lista transcript (supporto misto)
            def get_tx_list(gene_val, tx_val):
                txs = []
                if pd.notnull(tx_val) and tx_val != '.':
                    txs = [tx_val]
                else:
                    # supporto multi-gene separati da virgola
                    for g in str(gene_val).split(","):
                        g = g.strip()
                        txs += gene_transcripts.get(g, [])
                return txs

            tx_heads = get_tx_list(row[up_gene_col], row[up_trans_col])
            tx_tails = get_tx_list(row[dw_gene_col], row[dw_trans_col])

            for tx_head in tx_heads:
                for tx_tail in tx_tails:
                    try:
                        dna_head = extract_cds_sequence(
                            genome, gtf[tx_head], bp_head, "head",
                            mismatch=args.junction, tol=args.tol, intron=args.intron
                        )

                        dna_tail = extract_cds_sequence(
                            genome, gtf[tx_tail], bp_tail, "tail",
                            mismatch=args.junction, tol=args.tol, intron=args.intron
                        )

                    except Exception:
                        continue

                    fusion_dna = clean_dna(str(dna_head + dna_tail))

                    if translate:
                        orfs = run_orffinder_on_sequence(fusion_dna)
                        _, best_orf = get_longest_protein(orfs)
                        if not best_orf:
                            continue
                        seq = str(best_orf)
                    else:
                        seq = fusion_dna

                    # costruzione riga output
                    row_values = [str(row[c]) for c in df.columns]
                    if args.transcript_columns is None:
                        row_values += [tx_head, tx_tail]
                    row_values += [seq, str(len(seq))]

                    f.write("\t".join(row_values) + "\n")

    # leggi temp_file e salva finale
    output_df = pd.read_csv(temp_file, sep="\t")
    write_file(output_df, args.outfile)

"""
python extract_PLUS.py \
    --fasta_path /homes/vmelotti/project/data/genome/hg19.fa \
    --gtf_path /homes/vmelotti/project/data/genome/Homo_sapiens.GRCh37.87.gtf \
    --infile /homes/vmelotti/project/data/raw/data_all_SRR.pkl \
    --outfile /homes/vmelotti/project/data/raw/data_all_SRR_sequence.pkl \
    --transcript_columns transcript_id1,transcript_id2 \
    --breakpoint_columns bp1,bp2 \
    --gene_columns gene1,gene2 \
    --junction approximate

python extract_PLUS.py \
    --fasta_path /homes/vmelotti/project/data/genome/hg19.fa \
    --gtf_path /homes/vmelotti/project/data/genome/Homo_sapiens.GRCh37.87.gtf \
    --infile /homes/vmelotti/project/data/raw/extract_missing.tsv \
    --outfile //homes/vmelotti/project/data/raw/extract_missing_WITH_SEQ.tsv \
    --breakpoint_columns up_Genome_pos,dw_Genome_pos \
    --gene_columns up_gene,dw_gene \
    --junction approximate
"""