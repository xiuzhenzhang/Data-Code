

## Dataset Access 

Thank you for your interest in our research datasets. By downloading, accessing, or using this dataset, you agree to be bound by the terms and conditions outlined below. 

If you do not agree to these terms, you are not authorized to access or use the dataset.

### 1. Permitted Use
* **Non-Commercial Research Only:** This dataset is made available strictly for academic, educational, and scientific research purposes. 
* **Prohibited Commercial Use:** Any commercial use, including but not limited to product development, commercial services, internal business operations, or monetization through third parties, is strictly prohibited.

### 2. Restrictions on Redistribution
* **No Redistribution:** You may not copy, share, distribute, mirror, host, or transfer the dataset, any derived unencrypted subsets, or the access password to any third party, public repository, or external platform.
* **Access Control:** Other researchers within your institution or collaboration group must obtain their own password by submitting an official request through our designated access form.

### 3. Citation and Attribution
Any publications, presentations, reports, or software releases that utilize this dataset must properly cite our associated research publication(s) and acknowledge the dataset creators. 

*Please use the following citation format in your publications:*
>  "A Benchmark for LLM-Generated Disinformation Detection in SocialMedia", Submitted to NAACL-2027 

### 4. Termination
Your right to use this dataset terminates automatically if you breach any of these terms. Upon termination, you must immediately delete all copies of the dataset and associated files in your possession.

## Request

* 📝 Fill out the [Dataset Access Request Form](https://forms.cloud.microsoft/r/YQuBFMksjr). Once your request is reviewed and approved, information for accessing the dataset will be sent to your institutional email address.



## Code Details 

### 1. Create ML Datasets
We generate four dataset based ML datasets in three different settings: HU-Dis, AI-Dis and HU-AI-Dis. 

Create_ML_Datasets.py file creates these ML datasets using our generated data and existing IO-data from Seckin et al., "Lebeled datasets for research on information operations, 2025. 

### 2. Benchmark Code Details
DP_CoT_Baselines.py contains the Direct Prompt and CoT prompt-based experimental code for evaluating five LLM-based models across different datasets and experimental settings.
Transformer_Based_Baselines.py contains the code for evaluating seven Transformer-based models across different datasets and experimental settings.
X-troll_Baseline.py contains the implementation and evaluation of the existing X-Troll model, including its performance across different datasets and experimental settings.
