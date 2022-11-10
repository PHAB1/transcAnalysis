import pandas as pd
import glob

exp = open(snakemake.input[0],"r")
nc = snakemake.input[1]
rmats = snakemake.input[2]
red = snakemake.input[3]
output = snakemake.output[0]

for i in exp:
    exp = i
    break

#resume = pd.DataFrame()
file = open(output,"w")
file.write(exp)
file.close()
