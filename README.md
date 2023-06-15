# DeepT2: A deep learning model for type II polyketide natural product prediction
## Introduction 
DeepT2 utilizes deep learning techniques to identify type II polyketide (T2PK) synthases KSβ and their corresponding T2PK product within bacterial genomes. The method leverages ESM2 to transform KSβ sequences into embeddings, which are employed to train two separate classifiers using multi-layer perceptron for both KSβ and T2PKs classification with a consistency regularization-based semisupervised learning framework. In addition, the Mahalanobis distance-based algorithm was applied on each feature layer of the T2PK classifier to perform novelty detection and avoid the problem of overconfident softmax-based classifiers. Overall, our model could easily detect and classify KSβ either as a single sequence or metagenome input, and subsequently identify the corresponding T2PKs in a labeled categorized class or novel. 


![Biosynthesis-2_page-0001](https://github.com/Qinlab502/deept2/assets/117368489/670bb1b3-1cf7-4011-a114-f24cc47acc87)

## System requirment
We provide the DeepT2 conda environment, which can be built via ```conda env create -f environment_gpu.yml```.
The inclusion of the ```cudatoolkit==11.1.1``` dependency is deemed unnecessary in the absence of a GPU within the system configuration.
```conda env create -f environment_cpu.yml``` were optional.

## Pretrained language model
To run DeepT2, it requires the use of the pretrained protein language model ESM2. You can download the pretrained ESM2 model with 3B parameters using this link [(Link)](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt).

## Installation
### Ubuntu
You can acquire package throug ```git clone``` or directly download zip file:
```bash
git clone https://github.com/Qinlab502/deept2.git
```
Building conda environment:
```bash
conda env create -f environment_gpu.yml
```
Next, we have developed a Nextflow pipeline that allows users to easily execute DeepT2 prediction using either a bacteria genome or metagenome input:
```bash
nextflow run DeepT2.nf --genome "$PWD/genome.fasta" --outdir "$PWD/output" --prefix "Your sample"
```
Notably, the excuted DeepT2.nf should be placed in downloaded folder.\
Here is an example command for user reference:
```bash
nextflow run DeepT2.nf --genome "$PWD/example/example.fasta" --outdir "$PWD/output/example_result" --prefix "example"
```
And the prediction results will be saved in:
```bash
./output
```
We also provide the corresponding canonical prediction results in ```./example/example_result``` for your reference.

Tips: Nextflow will generate cache files that are stored in the ```work``` folder. Please feel free to delete this folder once the task has been completed.
## Dataset and model
We provide the datasets and the trained DeepT2 models here for those interested in reproducing our paper. The datasets used in this study are stored in ```./datasets/```. The trained DeepT2 models can be found under ```./model/```.

## Citation and contact
Citation:

Our work has been published in bioRxiv. Detailed information could be found in https://doi.org/10.1101/2023.04.18.537339

Contact:

Zhiwei Qin(z.qin@bnu.edu.cn)\
Jiaquan Huang(jiaquan_terry@bnu.edu.cn)
