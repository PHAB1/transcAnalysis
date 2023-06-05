library("DESeq2")

expr_matrix <- snakemake@input[['htseq']]
samples <- snakemake@params
output_cod <- snakemake@output[['out_cod']]

design_table <- as.data.frame(samples)
#design_table <- read.table(samples, header = TRUE,sep=",")

expr_matrix <- as.matrix(read.table(expr_matrix, col.names=sapply(design_table$sample, function(x) {tail(strsplit(x,"/",fixed=T)[[1]],n=1)}), row.names = 1))

design_table<-data.frame(conditions=design_table$group,row.names=colnames(expr_matrix))
design_table$conditions<-as.factor(design_table$conditions)

dds <- DESeqDataSetFromMatrix(countData=expr_matrix,colData=design_table,design= ~ conditions)
dds <- DESeq(dds)
dds_result <- as.data.frame(results(dds, contrast=c("conditions",as.vector(unique(design_table$conditions)))))

write.csv(dds_result,output_cod)
