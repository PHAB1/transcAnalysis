red <- snakemake@input[['red']]

groups <- snakemake@params[['groups']]
samples <- as.vector(snakemake@input[['samples']])

list_samples <- c()
for(i in samples) {
    list_samples <- append(list_samples,strsplit( strsplit(i, "/")[[1]][5], ".", fixed=T )[[1]][1])
}
samples <- list_samples
rm(list_samples)

output <- snakemake@output[[1]]

source(snakemake@params[["redits_file"]])

red <- read.table(red,sep=",",header=T)
p_list <- c()
for(r in 1:nrow(red)) {
    t <- rbind(red[samples][r,]*100, (1-red[samples][r,])*100 )
    p_list<-append(p_list,REDIT_LLR(data=as.matrix(round(t)), groups=groups)$p.value)
}

FDR_list <- p.adjust( p_list, method='BH')
red$FDR <- FDR_list

print(output)
write.csv(red, file = output, row.names = FALSE)
