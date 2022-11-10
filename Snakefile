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
        genome=config["files_path"]["genome_fa"]
    threads: workflow.cores
    output:
        directory("RED/")
    script:
        "scripts/reditools.py"

rule integration:
    input:
        exp = "expression/exp",
        nc_exp = "non-coding/exp",
        rmats = "rmats/",
        red = "RED/"
    conda:
        "envs/integration.yaml"
    output:
        "resume.csv"
    script:
        "scripts/resume.py"

