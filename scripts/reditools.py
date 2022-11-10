import os
import multiprocessing.dummy as mp

red_files = snakemake.input
genome = snakemake.params
output = snakemake.output

os.system("git clone https://github.com/tflati/reditools2.0.git")
os.system("mkdir %s"%output)

def do_reditools(i):
    f = i.split(".")[0]
    os.system("python2 reditools2.0/src/cineca/reditools.py -f %s -o RED/%s -r %s -s 2 -S -C"%(i,f,genome))

p=mp.Pool(snakemake.threads)
p.map(do_reditools,red_files)
p.close()
p.join()
    

