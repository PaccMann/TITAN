from Bio import SeqIO
from Bio import pairwise2
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    'directory',
    type=str,
    help='Directory containing V_segment_sequences.fasta and \
        J_segment_sequences.fasta files downloaded from IMGT'
)
parser.add_argument(
    'input_sequences',
    type=str,
    help='CSV or TSV file containing CDR3 info, V and J segment names.'
)
parser.add_argument(
    'v_seg_header', type=str, help='Header for column containing V segments.'
)
parser.add_argument(
    'j_seg_header', type=str, help='Header for column containing J segments.'
)
parser.add_argument(
    'cdr3_header', type=str, help='Header for column containing J segments.'
)
parser.add_argument(
    'output', type=str, help='Directory and filename of output csv file.'
)


def rename_Vseg(Vname):
    if Vname[1] == 'C':
        Vname = 'TRB' + Vname[4:]
        if Vname[Vname.find('-') + 1] == '0':
            Vname = Vname[:(Vname.find('-') +
                            1)] + Vname[(Vname.find('-') + 2):]
        if Vname[Vname.find('V') + 1] == '0':
            Vname = Vname[:(Vname.find('V') +
                            1)] + Vname[(Vname.find('V') + 2):]
    return Vname


def rename_Jseg(Jname):
    if Jname[1] == 'C':
        Jname = 'TRB' + Jname[4:]

        if Jname[Jname.find('-') + 1] == '0':
            Jname = Jname[:(Jname.find('-') +
                            1)] + Jname[(Jname.find('-') + 2):]
        if Jname[Jname.find('J') + 1] == '0':
            Jname = Jname[:(Jname.find('J') +
                            1)] + Jname[(Jname.find('J') + 2):]
    return Jname


def to_full_seq(directory, Vname, Jname, CDR3):
    ## Translate segment name into segment sequence
    foundV = False
    foundJ = False
    for Vrecord in SeqIO.parse(
        os.path.join(directory, 'V_segment_sequences.fasta'), "fasta"
    ):
        if type(Vname) != str or Vname == 'unresolved':
            print('Vname not string but ', Vname, type(Vname))
            Vseq = ''

        else:
            ## Deal with inconsistent naming conventions of segments
            Vname_adapted = rename_Vseg(Vname)
            if Vname_adapted in Vrecord.id:
                Vseq = Vrecord.seq
                foundV = True

    for Jrecord in SeqIO.parse(
        os.path.join(directory, 'J_segment_sequences.fasta'), "fasta"
    ):
        if type(Jname) != str or Jname == 'unresolved':
            print('Jname not string but ', Jname, type(Jname))
            Jseq = ''
        else:
            ## Deal with inconsistent naming conventions of segments
            Jname_adapted = rename_Jseg(Jname)
            if Jname_adapted in Jrecord.id:
                Jseq = Jrecord.seq
                foundJ = True

    if Vseq != '':
        ## Align end of V segment to CDR3
        alignment = pairwise2.align.globalxx(
            Vseq[-5:],  # last five amino acids overlap with CDR3
            CDR3,
            one_alignment_only=True,
            penalize_end_gaps=(False, False)
        )[0]
        best = list(alignment[1])

        ## Deal with deletions
        if best[0] == '-' and best[1] == '-':
            best[0] = Vseq[-5]
            best[1] = Vseq[-4]
        if best[0] == '-':
            best[0] = Vseq[-5]

        # remove all left over -
        best = "".join(list(filter(lambda a: a != '-', best)))
    else:
        best = CDR3

    ## Align CDR3 sequence to start of J segment
    if Jseq != '':
        alignment = pairwise2.align.globalxx(
            best,
            Jseq,
            one_alignment_only=True,
            penalize_end_gaps=(False, False)
        )[0]

        # From last position, replace - with J segment amino acid
        # until first amino acid of CDR3 sequence is reached
        best = list(alignment[0])[::-1]
        firstletter = 0
        for i, aa in enumerate(best):
            if aa == '-' and firstletter == 0:
                best[i] = list(alignment[1])[::-1][i]
            else:
                firstletter = 1

        # remove all left over -
        best = "".join(list(filter(lambda a: a != '-', best[::-1])))

    full_sequence = Vseq[:-5] + best

    return full_sequence, foundV, foundJ


def main(
    directory, input_sequences, v_seg_header, j_seg_header, cdr3_header, output
):
    name, extension = os.path.splitext(input_sequences)
    if extension == '.csv':
        seq_data = pd.read_csv(input_sequences)
    elif extension == '.tsv':
        seq_data = pd.read_csv(input_sequences, delimiter='\t')
    else:
        print('Please provide the input sequences in .tsv or .csv format.')
    full_seqs = []
    for i, row in seq_data.iterrows():
        Vname = row[v_seg_header]
        Jname = row[j_seg_header]
        CDR3 = row[cdr3_header]
        full_seq, foundV, foundJ = to_full_seq(directory, Vname, Jname, CDR3)
        full_seqs.append(full_seq)
    seq_data['full_seq'] = full_seqs
    seq_data.to_csv(output)


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    main(
        args.directory, args.input_sequences, args.v_seg_header,
        args.j_seg_header, args.cdr3_header, args.output
    )