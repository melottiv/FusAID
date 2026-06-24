from Bio import SeqIO
from Bio.Seq import Seq
import re
import matplotlib.pyplot as plt
import pandas as pd
from Bio import pairwise2  # modulo vecchio, ma quello nuovo non funziona bene... 
from collections import defaultdict
#from .run_orffinder import run_orffinder_on_sequence, get_longest_protein
import subprocess
import tempfile

def clean_dna(seq):
    return "".join([c for c in seq.upper() if c in "ATGCN"])


def run_orffinder_on_sequence(sequence_str):
    """
    Esegue ORFfinder su una sequenza fornita come stringa
    e restituisce l'output come stringa.
    """
    # Creiamo un file temporaneo per l'input
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".fasta") as temp_input, \
         tempfile.NamedTemporaryFile(mode='r+', suffix=".gff") as temp_output:

        # Scriviamo la sequenza nel file temporaneo in formato FASTA
        temp_input.write(">seq\n")
        temp_input.write(clean_dna(sequence_str))
        temp_input.flush()  # importante per far leggere il contenuto all'eseguibile

        # Costruiamo il comando ORFfinder
        cmd = ["./ORFfinder", "-in", temp_input.name,
               "-out", temp_output.name, "-outfmt", "0"]

        # Eseguiamo ORFfinder
        result=subprocess.run(cmd, check=True)
        
        if result.returncode != 0:
            print(result.stderr)
            print("FASTA file:", temp_input.name)
            raise RuntimeError("ORFfinder failed")

        # Torniamo all'inizio del file di output e leggiamo tutto
        temp_output.seek(0)
        return temp_output.read()

def get_longest_protein(orf_output):
    """
    Prende l'output FASTA di ORFfinder (amminoacidi) e restituisce
    l'header e la sequenza della proteina più lunga.
    """
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
    
    # aggiungi l'ultima sequenza
    if current_header is not None:
        sequences[current_header] = "".join(current_seq)

    # trova la sequenza più lunga
    if not sequences:
        return None, ""
    
    longest_header = max(sequences, key=lambda h: len(sequences[h]))
    return longest_header, str(sequences[longest_header])

#INFILE="combinedFGDB2genes_ORF_analyzed_gencode_h19v19_real_Inframe_only_transcript_seq_with_orffinder_result.txt"
INFILE= "/homes/vmelotti/project/data/raw/data_all_SRR.pkl"
#INFILE="/homes/vmelotti/project/data/raw/data_decider_positive.pkl"
#OUTFILE="/homes/vmelotti/project/data/raw/data_decider_positive_WITH_SEQUENCE.pkl"
OUTFILE="/homes/vmelotti/project/data/raw/data_all_SRR_WITH_SEQUENCE.pkl"
FASTA_FILE="/homes/vmelotti/project/data/genome/hg19.fa"
GTF_FILE="/homes/vmelotti/project/data/genome/Homo_sapiens.GRCh37.87.gtf"


# Tabella genetica standard NCBI (codoni -> amminoacidi)
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

def select_top_transcripts(gene_coords, gene_to_transcripts, max_transcripts=3):
    """
    Seleziona al massimo N transcript per ciascun gene,
    rimuovendo transcript con le stesse identiche coordinate CDS,
    ordinati per:
        1. lunghezza CDS totale
        2. numero di CDS exons
        3. posizione genomica (euristica tie-break)
    
    Se max_transcripts='all', prende tutti i transcript unici.
    """
    filtered_gene_coords = {}
    filtered_gene_to_transcripts = {}

    for gene, tx_list in gene_to_transcripts.items():
        ranking = []
        seen_cds_sets = set()  # traccia CDS già visti

        for tx in tx_list:
            if tx not in gene_coords:
                continue

            cds_exons = gene_coords[tx]["cds_exons"]
            if not cds_exons:
                continue

            # tuple immutabile per set
            cds_tuple = tuple(sorted(cds_exons))
            if cds_tuple in seen_cds_sets:
                continue  # skip duplicati
            seen_cds_sets.add(cds_tuple)

            total_len = sum(e[1] - e[0] + 1 for e in cds_exons)
            exon_count = len(cds_exons)
            leftmost = min(e[0] for e in cds_exons)  # tie-break

            ranking.append((tx, total_len, exon_count, leftmost))

        if not ranking:
            continue

        # ordina decrescente
        ranking.sort(key=lambda x: (x[1], x[2], -x[3]), reverse=True)

        # selezione dei transcript
        if str(max_transcripts).strip().lower() == 'all':
            selected_txs = [tx for tx, _, _, _ in ranking]
        else:
            selected_txs = [tx for tx, _, _, _ in ranking[:int(max_transcripts)]]

        filtered_gene_to_transcripts[gene] = selected_txs
        filtered_gene_coords[gene] = {tx: gene_coords[tx] for tx in selected_txs}

    return filtered_gene_coords, filtered_gene_to_transcripts



def best_alignment_identity(seq1, seq2):
    """
    Calcola l'identità percentuale del miglior allineamento locale
    tra due sequenze DNA o proteiche.
    Ritorna (percent_identity, aligned_seq1, aligned_seq2)
    """
    # Eseguo Smith-Waterman (local)
    alignments = pairwise2.align.localms(
        seq1, seq2,
        2,    # match score
        -1,   # mismatch penalty
        -5,   # gap opening
        -0.5  # gap extension
    )

    if not alignments:
        raise ValueError("Nessun allineamento trovato.")

    best = alignments[0]
    aligned_seq1 = best.seqA
    aligned_seq2 = best.seqB

    # Calcolo percent identity
    matches = sum(a == b for a, b in zip(aligned_seq1, aligned_seq2))
    length = sum(a != "-" and b != "-" for a, b in zip(aligned_seq1, aligned_seq2))

    if length == 0:
        return 0.0, aligned_seq1, aligned_seq2

    percent_identity = (matches / length) * 100
    return percent_identity, aligned_seq1, aligned_seq2


def reverse_complement(seq):
    complement = str.maketrans('ATCGatcg','TAGCtagc')
    return seq.translate(complement)[::-1]


def translate_seq(dna_seq, genetic_code=STANDARD_GENETIC_CODE):
    protein = ""
    for i in range(0,len(dna_seq)-2,3):
        codon = dna_seq[i:i+3].upper()
        protein += genetic_code.get(codon,'X')
    return protein




def select_orf(orfs):
    if not orfs:
        return None
    return max(orfs, key=lambda x: len(x['aa_seq']))


def find_orfs(dna_seq, min_len=75, start_codons=('ATG',), strand='both'):
    """
    Trova ORF in una sequenza DNA.
    dna_seq: sequenza DNA (stringa)
    min_len: lunghezza minima ORF in nucleotidi
    start_codons: tupla di codoni di inizio validi ('ATG','TTG',...)
    strand: 'plus','minus','both'
    ritorna: lista di dict {'strand','frame','start','end','nt_seq','aa_seq'}
    """
    results = []
    strands = []
    if strand == 'both':
        strands = ['plus','minus']
    else:
        strands = [strand]

    for s in strands:
        seq = dna_seq
        if s == 'minus':
            seq = reverse_complement(dna_seq)

        for frame in range(3):
            i = frame
            while i <= len(seq)-3:
                codon = seq[i:i+3]
                if codon.upper() in start_codons:
                    # trovato inizio ORF
                    start = i
                    protein = ""
                    j = i
                    while j <= len(seq)-3:
                        c = seq[j:j+3].upper()
                        aa = STANDARD_GENETIC_CODE.get(c,'X')
                        protein += aa
                        if aa == '*':  # stop
                            break
                        j += 3
                    end = j+3
                    if end-start >= min_len:
                        nt_seq = seq[start:end]
                        aa_seq = translate_seq(nt_seq)
                        results.append({'strand':s,'frame':frame+1,
                                        'start':start+1,'end':end,
                                        'nt_seq':nt_seq,'aa_seq':aa_seq})
                    i = j+3
                else:
                    i += 3
    return results


# -------------------------
# Carica il genoma in un dizionario {chromosome: Seq}
# -------------------------
def load_genome(fasta_path):
    genome = {}
    with open(fasta_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            genome[record.id] = record.seq
    return genome


# -------------------------
# Parsing GTF: estrai CDS, esoni e transcript_id
# -------------------------

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



def get_basic_transcripts(gtf_path):
    """
    Estrae i transcript_id che hanno:
      - tag "basic"
      - transcript_biotype "protein_coding"
    """
    basic_set = set()

    tag_regex = re.compile(r'tag "([^"]+)"')
    tx_regex = re.compile(r'transcript_id "([^"]+)"')
    biotype_regex = re.compile(r'transcript_biotype "([^"]+)"')

    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            feature = parts[2]
            if feature != "transcript":
                continue

            attributes = parts[8]

            # transcript_id
            tx_match = tx_regex.search(attributes)
            if not tx_match:
                continue
            tx_id = tx_match.group(1)

            # biotype (must be protein_coding)
            biotype_match = biotype_regex.search(attributes)
            if not biotype_match:
                continue
            if biotype_match.group(1) != "protein_coding":
                continue

            # tag (must include basic)
            tag_match = tag_regex.search(attributes)
            if not tag_match:
                continue

            tag_value = tag_match.group(1)
            tag_list = [t.strip() for t in tag_value.split(",")]

            if "basic" in tag_list:
                basic_set.add(tx_id)

    return basic_set

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


# -------------------------
# Estrai sequenza CDS
# -------------------------

def format_alignment(seq1, seq2, width=60):
    """
    Restituisce una stringa con l'allineamento visivo tra seq1 e seq2.
    | indica match, - gap, spazio mismatch.
    width: numero di basi per riga.
    """
    lines = []
    for i in range(0, len(seq1), width):
        s1 = seq1[i:i+width]
        s2 = seq2[i:i+width]
        mid = ''.join('|' if a == b and a != '-' else ' ' for a, b in zip(s1, s2))
        lines.append(f"Computed sequence:\t{s1}")
        lines.append("\t\t\t\t\t"+mid)
        lines.append(f"Correct sequence:\t{s2}")
        lines.append('')
    return '\n'.join(lines)


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


# -------------------------
# Esempio di utilizzo
# -------------------------
print("Loading genome and GTF...")
genome = load_genome(FASTA_FILE)
gtf = parse_gtf(GTF_FILE)
print("Done!")

ext=INFILE.split(".")[-1]
if ext == "pkl":
    input_file=pd.read_pickle(INFILE)
elif ext == 'csv':
    input_file=pd.read_csv(INFILE)
elif ext=="tsv":
    input_file=pd.read_csv(INFILE,sep='\t')

print(input_file.head())

input_file=input_file.head()

for idx,row in input_file.iterrows():
    tx_head=row['transcript_id1']
    tx_tail=row['transcript_id2']
    bp_head=int(row["bp1"])
    bp_tail=int(row["bp2"])
    print(f"Processing {tx_head}, {tx_tail}, {bp_head}, {bp_tail}")

    try:     
        dna_head = extract_cds_sequence(genome, gtf[tx_head], bp_head, "head",mismatch='approximate').upper()
        dna_tail = extract_cds_sequence(genome, gtf[tx_tail], bp_tail, "tail",mismatch='approximate').upper()
    except Exception as e:
        print(f"Error extracting sequences for transcripts {tx_head}, {tx_tail} at breakpoints {bp_head}, {bp_tail}: {e}\n")
        print(f"Error at row {idx}: {e}")
        continue

    fusion_dna =str(dna_head + dna_tail)
    orfs=run_orffinder_on_sequence(fusion_dna)
    header,best_orf=get_longest_protein(orfs)
    if not best_orf:
        continue
    cut_fusion=str(best_orf)

    print(f"found a {len(cut_fusion)} aa long sequence.")
    input_file.loc[idx, "amminoacid_sequence"] = cut_fusion
    input_file.loc[idx, "amminoacid_sequence_LENGTH"] = len(cut_fusion)

print(len(input_file))
input_file=input_file[input_file["amminoacid_sequence_LENGTH"]>1]
print(len(input_file))
print(f"Saving {len(input_file)} records")
input_file.to_pickle(OUTFILE)
