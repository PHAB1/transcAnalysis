import pandas as pd

samples_tsv = pd.read_csv(config["samples"])

samples = {}
samples["sample"] = list(samples_tsv["sample"])
samples["group"] = list(samples_tsv["group"])
samples["name"] = [x.split(".")[0] for x in samples]

def get_bam_input(wildcards):
    return samples["sample"]["wildcards.sample"]
