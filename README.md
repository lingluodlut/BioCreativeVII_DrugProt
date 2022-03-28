# NLM-NCBI team @ BioCreative VII DrugProt Track 
***
This repo contains the source code for extracting drug-protein relations in BioCreative VII DrugProt Track.
We propose a novel sequence labeling framework to the drug-protein relation extraction. Our method achieves the top performance in the BioCreative VII DrugProt Track.


## Dependency package
The codes have been tested using Python3.7 on CentOS and uses the following dependencies on a CPU and GPU:


- [TensorFlow 2.3.0](https://www.tensorflow.org/)
- [stanza 1.2.2](https://stanfordnlp.github.io/stanza/)
- [transformers 4.8.1](https://huggingface.co/docs/transformers/index)



## Model preparation

To run this code, you need to first download [the model file](https://ftp.ncbi.nlm.nih.gov/pub/lu/PhenoTagger/models.zip) ( the best single model, i.e., BioM-ELECTRAL with P->D), then unzip and put the model folder into the root folder.


## Extracting drug-protein relations from free text
<a name="tagging"></a>

You can use our trained model to extract drug-protein relations from biomedical texts by the *DrugProt_Tagging_PD.py* file.


The file requires 2 parameters:

- --input, -i, help="input file"
- --output, -o, help="output file to save the extracted relations"


The input file need to provide the text and named entity recognition (NER) information. There is one example in the */example/* folder.

Example:

```
$ python DrugProt_Tagging_PD.py -i ../example/example_input.tsv -o ../example/example_out.tsv
```

