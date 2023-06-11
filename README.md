# DeepT2: A deep learning model for type II polyketide natural product prediction
## Introduction 
DeepT2 utilizes deep learning techniques to identify type II polyketide (T2PK) synthases KSβ and their corresponding T2PK product within bacterial genomes. The method leverages ESM2 to transform KSβ sequences into embeddings, which are employed to train two separate classifiers using multi-layer perceptron for both KSβ and T2PKs classification with a consistency regularization-based semisupervised learning framework. In addition, the Mahalanobis distance-based algorithm was applied on each feature layer of the T2PK classifier to perform novelty detection and avoid the problem of overconfident softmax-based classifiers. Overall, our model could easily detect and classify KSβ either as a single sequence or metagenome input, and subsequently identify the corresponding T2PKs in a labeled categorized class or novel. 


![Biosynthesis-2_page-0001](https://github.com/Qinlab502/deept2/assets/117368489/670bb1b3-1cf7-4011-a114-f24cc47acc87)

## System requirment
python 3.8\
pytorch 2.0\
pandas 1.1.0\
fair-esm 2.0.0\
scikit-learn 1.2.2

## Pretrained language model
To run DeepT2, it requires the use of the pretrained protein language model ESM2. You can download the pretrained ESM2 model with 3B parameters using this link [(Link)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt).

## Run DeepT2 for prediction
For single genome input, we suggest prokka for CDS prediction:
```bash
prokka genome.fa --outdir ./output --prefix XX --kingdom Bacteria --rfam
```
However, the header of the generated FAA file should be simplified using the following command:
```bash
sed -i 's/ .*//' your_file.faa
```
Next, the FAA file should be converted to embedding when ESM2(3B) model was prepared:
```bash
python extract.py esm2_t36_3B_UR50D ./your_file.faa ./embedding --repr_layers 36 --include mean
```
Finally, simply run:
```bash
python DeepT2.py --fasta your_file.faa --embedding ./embedding --output ./results --name your_strain
```
And the prediction results will be saved in
```bash
./results
```
We also provide the corresponding canonical prediction results in ```bash ./example_output``` for your reference.
## Dataset and model
We provide the datasets and the trained DeepT2 models here for those interested in reproducing our paper. The datasets used in this study are stored in ```bash ./datasets/```. The trained LMDisorder models can be found under ```bash./model/```.

## Citation and contact
Citation:

Our work has been published in bioRxiv. Detailed information could be found in https://doi.org/10.1101/2023.04.18.537339

Contact:

Zhiwei Qin(z.qin@bnu.edu.cn)\
Jiaquan Huang(jiaquan_terry@bnu.edu.cn)
