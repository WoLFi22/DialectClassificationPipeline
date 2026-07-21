# Speech-Audio Classification Pipeline

This repository provides a pipeline for dialect classification using deep learning on raw audio files. The pipeline utilizes a Google model to extract embeddings from audio segments of uniform length, which are subsequently classified using a Multilayer Perceptron (MLP).

In addition to the classification pipeline, the repository contains methods for
deriving and analyzing a continuous dialectality measure from the learned audio
embeddings.

### Key Features

- **Easily adjustable parameters:** The main parameters are configurable, allowing for repeated use of the pipeline and facilitating experimentation.
- **Significance testing:** The repository includes a notebook for examining  significant differences between experimental runs.
- **Results visualization:** Evaluation results can be exported as graphical summaries.
- **User-friendly execution:** The complete classification pipeline can be executed through the main pipeline notebook.
- **GPU support:** The pipeline is designed to use GPU acceleration.
- **Continuous dialectality analysis:** The `vertical` workflow implements the
  Dialect Distance Measure from Classification Embeddings (DIME) and subsequent
  speaker-level and segment-length analyses.

### Getting Started
- Adjust Parameters: Configure the key parameters in the '_00_Pipeline' file to suit your experiment.
- Run the Pipeline: Execute the '_00_Pipeline' notebook to process the audio data, perform classification, and visualize results.
- Analyze Results: Once the pipeline has finished running, you only need to analyze the results. :wink:

This pipeline simplifies the exploration of dialectal differences, making research more efficient and effective.

### Contents

- [Speech-Audio Classification Pipeline](#speech-audio-classification-pipeline)
    - [Key Features](#key-features)
    - [Getting Started](#getting-started)
    - [Contents](#contents)
  - [Requirements](#requirements)
    - [Main Dependencies](#main-dependencies)
    - [Additional Dependencies for Preprocessing and Augmentation](#additional-dependencies-for-preprocessing-and-augmentation)
    - [GPU Support](#gpu-support)
    - [Installation](#installation)
  - [Input Audio Folder Structure](#input-audio-folder-structure)
  - [Audio Specifications](#audio-specifications)
  - [Usage Instructions](#usage-instructions)
  - [Pretrained Model](#pretrained-model)
    - [Model Details](#model-details)
    - [Using the Pretrained Model](#using-the-pretrained-model)
  - [Dialect Distance Measure from Classification Embeddings (DIME)](#dialect-distance-measure-from-classification-embeddings-dime)
    - [DIME Workflow](#dime-workflow)
    - [Notebooks](#notebooks)
    - [Phonetic Reference: D-values](#phonetic-reference-d-values)


## Requirements

This pipeline has been tested with the following versions:

### Main Dependencies
- ipynb                        0.5.1
- keras                        2.10.0
- librosa                      0.9.2
- matplotlib                   3.5.2
- numpy                        1.22.4
- pandas                       1.4.3
- praat-parselmouth            0.4.3
- pydub                        0.25.1
- python                       3.9.12
- scipy                        1.8.1
- seaborn                      0.11.2
- tensorflow                   2.10.0
- tensorflow-hub               0.13.0

### Additional Dependencies for Preprocessing and Augmentation
- audiomentations              0.30.0
- noisereduce                  3.0.0
- praat (program)              6.2.14
- pyloudnorm                   0.1.1
- soundfile                    0.12.1

### GPU Support
For GPU support, you will need:
- An Nvidia GPU (tested with <em>NVIDIA GeForce RTX 3080 Ti Laptop GPU</em>)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (tested with version 11.2)
- [cuDNN](https://developer.nvidia.com/cudnn) (tested with version 8.9.3)

For compute capability for your Nvidia GPU see [here](https://developer.nvidia.com/cuda-gpus).


### Installation

To install the required packages, you can use the following command:

```bash
pip install -r requirements.txt
```

## Input Audio Folder Structure

<em>Paths are currently only working on Windows-Systems!</em>

This repository follows a specific structure for organizing audio data:
- **Audio Folder**: All audio files should be stored in a designated folder.
- **Subfolders**: Within the audio folder, there should be subfolders named after the classes (dialects) of the audio data.
- **Speaker Subfolders**: Inside each class subfolder, create subfolders for the individual speakers from whom the audios are collected.
- **Single Audio per Speaker**: Each speaker subfolder should contain only one audio file, ensuring clarity and simplicity in data organization.

```
Audio_Folder/
│
├── Class_1/
│   ├── Speaker_A/
│   │   └── audio_1.wav
│   └── Speaker_B/
│       └── audio_2.wav
│
└── Class_2/
    ├── Speaker_C/
    │   └── audio_3.wav
    └── Speaker_D/
        └── audio_4.wav

```

## Audio Specifications

To ensure that the pipeline operates correctly, the audio files should have the following properties:
- **Channels:** mono
- **Sampling rate:** 16 kHz
- **Bit depth:** 16 bit

For preprocessing the audio data accordingly, refer to the 'Preprocessing' notebook provided in this repository. It contains instructions and code for any necessary preprocessing steps.

## Usage Instructions

In this section, you'll find detailed instructions on how to effectively utilize the main file of this repository. Follow these steps to get started with the provided functionality.

In the following graphic, you can see an image of the entire pipeline. It is divided into its individual steps. The gray dashed blocks each represent a notebook. Under these blocks, you will find the name of the notebook and the most important parameters, which can be adjusted in the `_00_Pipeline.ipynb` file.

![Pipeline](https://github.com/user-attachments/assets/89334372-4c31-4b50-bdb3-c16e9e6b7a46)


All other parameters that can be adjusted are also listed in the `_00_Pipeline.ipynb` file. Once all parameters are correctly filled in, the `_00_Pipeline.ipynb` notebook can be executed.

Additionally, the most important functions in the individual notebooks are described with their parameters in the notebooks itself, even if these do not need to be changed for execution.

## Pretrained Model

This repository provides a pretrained model that can be used for inference or as initialization for further training.
The model corresponds to the experimental setup described and used in *[Paper XY]*.

### Model Details
- **File:** `model_weights_0.h5`
- **Training data:**
  - 18 German dialect classes
  - more than 42 hours of speech data
  - multiple speaker generations
- **Input representation:**
  - 10-second segments
  - 16 kHz
  - mono audio

### Using the Pretrained Model

To run the pipeline with the provided pretrained weights, adjust the following parameters in `_00_Pipeline.ipynb`:

- Set the dense layer size to match the pretrained model:

```python
# units dense layer
units = 512
```

- Enable test-only mode to skip training:

```python
# when True the Model makes predictions on Audios in 'data_path_test'
test_only = True
```

## Dialect Distance Measure from Classification Embeddings (DIME)

This repository includes an extension of the pipeline that derives a continuous dialectality score from the generated speech embeddings.

Unlike the main classification task, which predicts discrete dialect classes, DIME models dialectal variation as a **continuous value** based on the embeddings produced by the pipeline.  

### DIME Workflow

The workflow uses embeddings from three recording conditions:

- **WSS:** Wenker sentences spoken in Standard German;
- **WSD:** Wenker sentences translated into and spoken in the local dialect;
- **FG:** free conversations.

Embeddings from an external Standard German dataset are used to estimate a
standard-language reference center. A discriminative direction separating WSS
and WSD is learned using logistic regression.

Each embedding is represented by two complementary components:

1. **Projection-based component:** its signed position, relative to the external
   Standard German center, along the discriminative WSS–WSD direction.
2. **Distance-based component:** its distance from the external Standard German
   center.

Both components are z-normalized relative to the external Standard German
embeddings and combined into the final DIME score:

```text
DIME(x) = w · z_dist(x) + (1 − w) · z_proj(x)
```

The weight `w` is selected by maximizing the correlation between aggregated
DIME scores and external D-values.

The following figure summarizes the complete DIME workflow, including audio
segmentation, embedding extraction, estimation of the discriminative WSS–WSD
direction, centering relative to the external Standard German reference,
calculation of projection and distance scores, z-normalization, and weighted
score combination.

<!-- Insert DIME workflow figure here -->

*Figure: Overview of the computation of segment-level DIME scores.*

### Notebooks

- **`01_EmbeddingDialectScore.ipynb`** computes segment-level DIME scores. It estimates the external Standard German center and the discriminative WSS–WSD direction, calculates the projection- and distance-based components, selects their weight using D-values, and evaluates the resulting scores.

- **`02_speaker_types.ipynb`** analyzes DIME scores at the speaker level. It derives speaker-specific WSS and WSD reference ranges, examines DIME trajectories in free conversations, and classifies speakers according to their observed repertoire and shifting or switching behavior.

- **`segment_length_evaluation.ipynb`** compares DIME scores obtained with different segment lengths and overlap configurations. The comparison includes correlations with D-values, WSS–WSD discrimination, local score variability, mixed-effects estimates, and agreement between configurations.

### Phonetic Reference: D-values

D-values are used as an external phonetic reference for evaluating the embedding-based dialectality scores.

D-values quantify dialectality as the average phonetic distance of a speech sample to a codified standard pronunciation at the word level. They are based on narrow phonetic transcriptions and a rule-based comparison to standard forms.

They therefore provide a linguistically grounded measure of dialectal variation, independent of machine learning models.

The file `data/d_values.csv` contains the D-values used in this repository.

Further information on D-values:
- https://rede-infothek.dsa.info/?page_id=211
- https://www.regionalsprache.de/SprachGIS/