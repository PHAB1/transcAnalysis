import os
import multiprocessing.dummy as mp

red_files = snakemake.input
genome = snakemake.params
output = snakemake.output

os.system("git clone https://github.com/jumphone/SPRINT.git")
os.system("mkdir -p %s"%output)

def do_reditools(i):
    f = (i.split("/")[-1]).split(".")[0]
    os.system("python3 reditools2.0/src/cineca/reditools.py -f %s -o %s/%s -r %s -S"%(i,output,f,genome))

p=mp.Pool(snakemake.threads)
p.map(do_reditool,sred_files)
p.close()
p.join()
    

