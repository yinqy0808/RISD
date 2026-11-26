#!/usr/bin/env python3
import csv


def read_fasta(filename):
    """Read FASTA records as (id, seq), preserving input order."""
    records = []
    current_id = None
    current_seq = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            records.append((current_id, "".join(current_seq)))
    return records


def read_csv(filename):
    """Read filename and seq columns from a CSV file."""
    records = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["filename"]
            seq = row["seq"]
            if name.endswith(".pdb"):
                name = name[:-4]
            records.append((name, seq))
    return records


def remove_duplicate_sequences(records):
    """Remove duplicate sequences while keeping the first record."""
    seen = set()
    unique = []
    for id_, seq in records:
        if seq not in seen:
            seen.add(seq)
            unique.append((id_, seq))
    return unique


def write_fasta(records, output_file):
    """Write FASTA records."""
    with open(output_file, "w") as f:
        for id_, seq in records:
            f.write(f">{id_}\n")
            for i in range(0, len(seq), 1000):
                f.write(seq[i:i + 1000] + "\n")


def main():
    fasta_file = "./9-7316.fasta"
    csv_file = "./diff_all.csv"
    output_file = "9-7316_merged.fasta"

    fasta_records = read_fasta(fasta_file)
    csv_records = read_csv(csv_file)
    all_records = fasta_records + csv_records
    unique_records = remove_duplicate_sequences(all_records)

    write_fasta(unique_records, output_file)
    print(f"Done. Kept {len(unique_records)} unique sequences in {output_file}")


if __name__ == "__main__":
    main()
