import os
import pandas as pd
import pyranges as pr
import multiprocessing.dummy as mp

# input
red_files = snakemake.input

# params
genome = snakemake.params[0]
samples = snakemake.params[1]
samples = samples["group"]
gtf = snakemake.params[2]
min_reads = 20
min_frac = 0.05
min_editions = 10
ed_types = ["AG","TC","GA","CT","GC","CG","TG","CA","AC","GT","TA","AT"]

# Output
output = snakemake.output[0]
red_all_output = snakemake.output[1]

os.system("git clone https://github.com/BioinfoUNIBA/REDItools2.git")
os.system("git clone https://github.com/gxiaolab/REDITs.git") # REDITs - Statistical module
os.system("mkdir -p %s"%output)

def do_reditools(i):
    f = (i.split("/")[-1]).split(".")[0]
    os.system("python3 REDItools2/src/cineca/reditools.py -f %s -o %s/%s -r %s -S"%(i,output,f,genome))
    
p=mp.Pool(snakemake.threads)
p.map(do_reditools,red_files)
p.close()
p.join()

RedDist = pd.DataFrame(index=ed_types)
RedFreqMat = pd.DataFrame(columns=["Region","Position"])
for i in red_files:
    f = (i.split("/")[-1]).split(".")[0]
    red_count = pd.read_csv("%s/%s"%(output,f),sep="\t")
    red_count = red_count.query("`Coverage-q30` > 20 and Frequency > 0.05 and Frequency < 0.95 and `Coverage-q30`*Frequency >= 10 and AllSubs.str.len() < 3") # filtered
    
    RedDist = RedDist.merge(red_count["AllSubs"].value_counts(), left_index=True, right_index=True) # Distribution

    red_count = red_count[["Region","Position","Frequency"]] 
    red_count.columns = ["Region","Position",f]
    RedFreqMat = RedFreqMat.merge(red_count,  how='outer', left_on=['Region','Position'], right_on = ['Region','Position'])

gtf = pr.read_gtf(gtf) 
gtf = gtf.df

RedFreqMat.to_csv(red_all_output)

RedDist.columns = samples
RedDist.to_csv("%s/RedDist.csv"%output)

