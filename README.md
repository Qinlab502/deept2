# DeepT2: A deep learning model for type II polyketide natural product prediction
## Introduction 
DeepT2 utilizes deep learning techniques to identify type II polyketide (T2PK) synthases KSβ and their corresponding T2PK product within bacterial genomes. The method leverages ESM2 to transform KSβ sequences into embeddings, which are employed to train two separate classifiers using multi-layer perceptron for both KSβ and T2PKs classification. In addition, out model could easily detect and classify KSβ either as a single sequence or metagenome input, and subsequently identify the corresponding T2PKs in a labeled categorized class or as novel.


![Biosynthesis-2_page-0001](https://github.com/Qinlab502/deept2/assets/117368489/670bb1b3-1cf7-4011-a114-f24cc47acc87)

## System requirment
python 3.8
pytorch 2.0
pandas 1.1.0
esm2 3B
scikit-learn 1.2.2

##Pretrained language model
Pretrained protein language model ESM2 are needed to run DeepT2: Downloading the pretrained ESM2 model with 3B parameters (Link(https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt)).
