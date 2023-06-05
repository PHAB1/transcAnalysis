import pandas as pd
import glob
import os

samples = snakemake.input["samples"]

# rmats files
rmats = snakemake.input["rmats"] # Directory

# A3SS -> Obtein and count significant
A3SS = pd.read_csv(os.path.join(rmats,"A3SS.MATS.JCEC.txt"),sep="\t")
A3SS =  A3SS[(A3SS['FDR'] < 0.05) & (abs(A3SS['IncLevelDifference']) > 0.1)]["GeneID"].value_counts()

# A5SS -> Obtein, count significant and merge
A5SS = pd.read_csv(os.path.join(rmats,"A5SS.MATS.JCEC.txt"),sep="\t")
A5SS =  A5SS[(A5SS['FDR'] < 0.05) & (abs(A5SS['IncLevelDifference']) > 0.1)]["GeneID"].value_counts()
resume = pd.merge(A3SS, A5SS, left_index=True, right_index=True,how='outer').fillna(0)

# RI -> Obtein, count significant and merge
RI = pd.read_csv(os.path.join(rmats,"RI.MATS.JCEC.txt"),sep="\t")
RI = RI[(RI['FDR'] < 0.05) & (abs(RI['IncLevelDifference']) > 0.1)]["GeneID"].value_counts()
resume = pd.merge(resume, RI, left_index=True, right_index=True,how='outer').fillna(0)

# MXE -> Obtein, count significant and merge
MXE = pd.read_csv(os.path.join(rmats,"MXE.MATS.JCEC.txt"),sep="\t")
MXE = MXE[(MXE['FDR'] < 0.05) & (abs(MXE['IncLevelDifference']) > 0.1)]["GeneID"].value_counts()
resume = pd.merge(resume, MXE, left_index=True, right_index=True,how='outer').fillna(0)

# SE -> Obtein, count significant and merge
SE = pd.read_csv(os.path.join(rmats,"SE.MATS.JCEC.txt"),sep="\t")
SE = SE[(SE['FDR'] < 0.05) & (abs(SE['IncLevelDifference']) > 0.1)]["GeneID"].value_counts()
resume = pd.merge(resume, SE, left_index=True, right_index=True,how='outer').fillna(0)

# expression -> Obtein, count significant and merge
exp = pd.read_csv(snakemake.input["exp"],index_col=0)
exp = exp[(exp['padj'] < 0.05) & (abs(exp['log2FoldChange']) > 1)]
exp = exp["log2FoldChange"]
resume = pd.merge(resume, exp, left_index=True, right_index=True,how='outer').fillna(0)

# non-coding -> Obtein, count significant and merge
nc = pd.read_csv(snakemake.input["nc_exp"],index_col=0)
nc = nc[(nc['padj'] < 0.05) & (abs(nc['log2FoldChange']) > 1)]
nc = nc["log2FoldChange"]
resume = pd.merge(resume, nc, left_index=True, right_index=True,how='outer').fillna(0)

# RED - > Obtein, count significant and merge
red = pd.read_csv(snakemake.input["red"], index_col="gene_id")
red = red.fillna(0)
red = red[red['FDR'] < 0.05]
red = red.index.value_counts()
resume = pd.merge(resume, red, left_index=True, right_index=True,how='outer').fillna(0)
resume.columns = ["A3SS","A5SS","RI","MXE","SE","mRNA_FC","lncRNA_FC","RED"]

output = snakemake.output[0]

resume.to_csv(output)
