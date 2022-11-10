samples = snakemake.input
params = snakemake.params[0]
output = snakemake.output

f1 = open("%s"%output[0],"w")
f2 = open("%s"%output[1],"w")

g1, g2 = [],[]
for i in range(len(samples)):
    if params[i] == params[0]:
        g1.append(samples[i])
    else:
        g2.append(samples[i])

f1.write("%s"%(",".join(g1)) )
f2.write("%s"%(",".join(g2)) )

f1.close()
f2.close()
