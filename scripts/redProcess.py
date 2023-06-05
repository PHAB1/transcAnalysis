from pybedtools import BedTool
import pandas as pd
import glob
import os
import tempfile

# rmats files
red = pd.read_csv(snakemake.input["red"], index_col=0)
red = red.fillna(0)

# Params
gtf = snakemake.params[0]

# Output
output = snakemake.output[0]

# Configure red columns
red['end'] = red["Position"] + 1
red_columns = ["Region","Position","end"] + red.columns.tolist()[2:]
red = red[red_columns]

# Rename columns name
#exp = exp.rename(columns={exp.columns[0]: 'ensembl'})
#nc = nc.rename(columns={nc.columns[0]: 'ensembl'})

#file = open(output,"w")
#file.write(exp)
#file.close()

from gtfparse import read_gtf

gtf = read_gtf(gtf)
gtf_columns = ["seqname","start","end","gene_id"]
gtf = gtf[gtf_columns]

import pybedtools

with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file1, \
     tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file2:
    red.to_csv(temp_file1.name,sep="\t", index=False,header=False)
    gtf.to_csv(temp_file2.name,sep="\t", index=False,header=False)

    # Get the paths of the temporary files
    temp_file_path1 = temp_file1.name
    temp_file_path2 = temp_file2.name
#red.to_csv("red.bed",sep="\t",index=False,header=False)
#gtf.to_csv("gtf.bed",sep="\t",index=False,header=False)

red_bed = pybedtools.BedTool(temp_file_path1)
gtf_bed = pybedtools.BedTool(temp_file_path2)

red_gtf = red_bed.intersect(gtf_bed,wa=True,wb=True).moveto(os.path.join(output))
red_ensembl = pd.read_csv(output,sep="\t",header=None)
red_ensembl.columns = red_columns + gtf_columns

red_ensembl.to_csv(output)
