# transcAnalysis Pipeline

The __transcAnalysis__ pipeline is a comprehensive tool that allows for the analysis of transcriptome data, including differential expression, alternative splicing, lncRNA and RNA editing analysis, with a specific focus on A-to-I editing mediated by the ADAR protein. This tutorial will guide you through the process of setting up and running the transcAnalysis pipeline using Snakemake.
Prerequisites

Before you start, make sure you have the following prerequisites installed on your system:

- Anaconda3 (v23.1.0)
- Snakemake (v7.25.0)

## Usage

The __transcAnalysis__ pipeline performs mRNA, lncRNA, AS, and RED expression analysis from a BAM file created after the alignment of RNA-seq reads from fastq files. The pipeline facilitates the analysis by allowing the execution of each step with only one command line, which is possible due to the use of the Snakemake pipeline manager. Before using the transcAnalysis pipeline, it is necessary to have Conda installed and activated.

To Download the __transcAnalysis__ pipeline, you need to clone the repository from GitHub:

```
git clone https://github.com/PHAB1/transcAnalysis.git
```

To run the pipeline, use the following command:

```
cd transcAnalysis

snakemake --cores [NUMBER OF CORES] --configfile [CONFIG FILE]
```
<sub>Note that it is necessary to edit the config and the samples.tsv file before running the pipeline.</sub>
