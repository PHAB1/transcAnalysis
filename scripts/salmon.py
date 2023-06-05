import os
import multiprocessing.dummy as mp

samples = snakemake.input
genome_fa = snakemake.params[0]
genome_gtf = snakemake.params[1]
output = snakemake.output

os.system("mkdir %s"%output)

print(genome_fa,genome_gtf)
def do_salmon(i):
    f = (i.split("/")[-1]).split(".")[0]
    os.system("salmon quant -t %s -l A -a %s -g %s -o %s/%s"%(genome_fa,i,genome_gtf,output,f))

p=mp.Pool(snakemake.threads)
p.map(do_salmon,samples)
p.close()
p.join()
    

