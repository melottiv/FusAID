import subprocess
import tempfile

import re

def evaluate_orf_quality(header, sequence):
    """
    Classifica un ORF come buono, mediocre o scadente in base a lunghezza e posizione.
    """
    # Estrazione delle coordinate start e end dal nome dell'ORF
    match = re.search(r'(\d+):(\d+)', header)
    if not match:
        return "Non valido", 0
    
    start, end = map(int, match.groups())
    orf_length = end - start
    
    # Determina se è parziale
    partial_flag = "partial" in header
    
    # Criteri di selezione: lunghezza e posizione
    if partial_flag:
        return "Parziale", orf_length

    # Valutazione della lunghezza
    if orf_length < 100:
        return "Scadente", orf_length
    elif orf_length < 300:
        return "Mediocre", orf_length
    else:
        # Gli ORF centrali tendono ad essere preferiti
        sequence_length = len(sequence)
        # Verifica che l'ORF non sia troppo vicino ai bordi
        if start < 0.1 * sequence_length or end > 0.9 * sequence_length:
            return "Mediocre", orf_length
        else:
            return "Buono", orf_length

def get_best_orf(orf_output, sequence):
    """
    Estrae e seleziona il miglior ORF (quello con la lunghezza massima e più centrale) 
    da un output di ORFfinder.
    """
    sequences = {}
    current_header = None
    current_seq = []

    # Analizza ogni linea dell'output ORFfinder
    for line in orf_output.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith(">"):  # Nuovo ORF
            if current_header is not None:
                sequences[current_header] = "".join(current_seq).replace(" ", "")
            current_header = line
            current_seq = []
        else:
            current_seq.append(line)
    
    # Aggiungi l'ultima sequenza
    if current_header is not None:
        sequences[current_header] = "".join(current_seq).replace(" ", "")
    
    # Se non ci sono sequenze, ritorna None
    if not sequences:
        return None, ""

    # Ora seleziona il miglior ORF
    best_orf_header = None
    best_orf_quality = None
    best_orf_length = 0
    for header, seq in sequences.items():
        quality, length = evaluate_orf_quality(header, sequence)
        if best_orf_quality is None or quality == "Buono" and length > best_orf_length:
            best_orf_header = header
            best_orf_quality = quality
            best_orf_length = length

    return best_orf_header, sequences.get(best_orf_header, "")

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

