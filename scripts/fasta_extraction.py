import argparse

def extract_fasta_sequences(header_file, fasta_file, output_file):
    headers = []
    with open(header_file, 'r') as f:
        headers = [line.strip() for line in f]

    fasta_sequences = {}
    header = ""
    with open(fasta_file, 'r') as f:
        for line in f:
            if line[0] == ">":
                header = line[1:].strip()
                fasta_sequences[header] = ""
            else:
                fasta_sequences[header] += line.strip()

    with open(output_file, 'w') as f:
        for header in headers:
            if header in fasta_sequences:
                f.write(">" + header + "\n")
                for i in range(0, len(fasta_sequences[header]), 60):
                    f.write(fasta_sequences[header][i:i+60] + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract FASTA sequences based on headers.')
    parser.add_argument("--header", type=str, help='Path to the header file')
    parser.add_argument("--input", type=str, help='Path to the FASTA file')
    parser.add_argument("--output", type=str, help='Path to the output file')

    args = parser.parse_args()
    extract_fasta_sequences(args.header, args.input, args.output)
