library("dplyr")

rawCounts = snakemake@input[['htseq_cod']]
sampleData = snakemake@params[['sampleData']]

print(rawCounts)
print(sampleData)
