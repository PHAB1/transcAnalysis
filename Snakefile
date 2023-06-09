import pandas as pd

configfile: "config.yaml"

include: "rules/common.smk"

rule all:
    input:
        "resume.csv"

rule preprocess:
    input:
        samples["sample"]
    output:
        "f1.rmats",
        "f2.rmats"
    params:
        groups=samples["group"]
    script:
        "scripts/preprocess.py"

rule htseq:
    input:
        samples["sample"],
        gtf=config["files_path"]["genome_reg"]
    params:
        stranded=expand("{genome}", genome=config["params"]["stranded"])
    conda:
        "envs/htseq.yaml"
    output:
        "expression/exp"
    shell:
        "htseq-count -f bam -r pos {input} > {output}"

rule htseq_nc:
    input:
        samples["sample"],
        gtf=config["files_path"]["genome_reg"]
    params:
        stranded=expand("{genome}", genome=config["params"]["stranded"])
    conda:
        "envs/htseq.yaml"
    output:
        "non-coding/exp"
    shell:
        "htseq-count -f bam -r pos {input} --additional-attr=gene_type > {output}"

rule dif_exp:
    input:
        htseq="expression/exp"
    params:
        sampleData=samples
    conda:
        "envs/deseq2.yaml"
    output:
        out_cod="expression/dif_exp.csv"
    script:
        "scripts/expression.R"

rule dif_exp_nc:
    input:
        htseq="non-coding/exp"
    params:
        sampleData=samples
    conda:
        "envs/deseq2.yaml"
    output:
        out_cod="non-coding/dif_exp.csv"
    script:
        "scripts/expression.R"

'''
rule salmon:
    input:
        samples["sample"]
    conda:
        "envs/salmon.yaml"
    params:
        genome_fa=config["files_path"]["genome_fa"],
        genome_gtf=config["files_path"]["genome_reg"]
    threads: workflow.cores
    output:
        directory("salmon_out/")
    script:
        "scripts/salmon.py"
'''

rule rmats:
    input:
       f1="f1.rmats",
       f2="f2.rmats",
       gtf=config["files_path"]["genome_reg"]
    params:
        read_len = config["params"]["read_len"]
    conda:
        "envs/rmats.yaml"
    output:
        directory("rmats/")
    shell:
        "rmats.py --b1 {input.f1} --b2 {input.f2} --gtf {input.gtf} -t paired --readLength {params.read_len} --nthread 4 --od {output} --tmp .tmp --variable-read-length --allow-clipping --novelSS"

rule reditools:
    input:
        samples["sample"]
    conda:
        "envs/reditools.yaml"
    params:
        genome=config["files_path"]["genome_fa"],
	sample=samples,
	gtf=config["files_path"]["genome_reg"]
    threads: workflow.cores
    output:
        directory("RED/"),
	"RED/RedFreqMat.csv"
    script:
        "scripts/reditools.py"

rule redProcess:
    input:
        red = "RED/RedFreqMat.csv",
	#salmon = "salmon_out/"
    conda:
        "envs/resume.yaml"
    params:
        genome=config["files_path"]["genome_reg"]
    output:
        temp("RED/red_ensembl_temp.csv")
    script:
        "scripts/redProcess.py"

rule redits:
    input:
        red="RED/red_ensembl_temp.csv",
	samples=samples["sample"]
    conda:
        "envs/resume.yaml"
    params:
        groups=samples["group"],
        redits_file = "REDITs/REDIT_LLR.R"
    output:
        "RED/red_ensembl.csv"
    script:
        "scripts/redits.R"

'''
rule SPRINT:
    input:
         samples["sample"]
    conda:
        "envs/sprint.yaml"
    params:
        genome=config["files_path"]["genome_fa"]
    threads: workflow.cores
    output:
        directory("RED")
    script:
        "scripts/SPRINT.py"
'''

rule integration:
    input:
        exp = "expression/dif_exp.csv",
        nc_exp = "non-coding/dif_exp.csv",
        rmats = "rmats/",
        red = "RED/red_ensembl.csv",
	samples = samples["sample"]
	#salmon = "salmon_out/"
    conda:
        "envs/resume.yaml"
    params:
    	expFDR_threshold=config["params"]["expFDR_threshold"],
    	foldChange_threshold=config["params"]["foldChange_threshold"],
	AS_FDR_threshold=config["params"]["AS_FDR_threshold"],
	AS_incLevel_threshold=config["params"]["AS_incLevel_threshold"],
	RED_FDR_threshold=config["params"]["RED_FDR_threshold"]
    output:
        "resume.csv"
    script:
        "scripts/resume.py"

