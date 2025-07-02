# smolvlm-vqa

## Introduction

This project explores visual-language modeling and interpretability using the [SmolVLM](https://huggingface.co/papers/2504.05299) family of models, specifically the SmolVLM-Instruct and SmolVLM-Base variants. 
It leverages two datasets: the [merve/vqav2-small](https://huggingface.co/datasets/merve/vqav2-small) dataset for visual question answering tasks and the elements dataset based on [this paper](https://arxiv.org/pdf/2404.03713v1) containing images annotated with object properties like color, shape, and texture. 
The project performs visual-language inference to generate descriptive answers, and uses embedding-based analyses including attention heatmap visualizations and PCA projections of vision patch embeddings to interpret how the models relate image regions to text inputs. The resulting visualizations provide insights into model focus and patch-level representations for both natural images and object datasets.

## Getting started

Create environment to install the relevant packages using:

`conda env create -f smolvlm.yml --name smolvlm`

`conda activate smolvlm`

## The scripts folder

- `merve_attention.py` - Generates attention heatmaps by visualizing the similarity between visual patches and question embeddings using the `SmolVLM-256M-Instruct` model on the `merve/vqav2-small` dataset.
- `infer_elements_instruct.py` - Runs visual-language inference using the `SmolVLM-Instruct` model on `elements` dataset to describe object properties (color, shape, texture) in images and saves the outputs alongside ground-truth labels to a CSV file.
- `infer_elements_base.py` - Performs visual-language inference with the `SmolVLM-Base` model on `elements` dataset to generate descriptive outputs for images and saves the results alongside ground-truth labels.
- `pca_merve_instruct.py` - Extracts vision patch embeddings from SmolVLM’s vision encoder on `merve/vqav2-small` images and visualizes them using PCA alongside patch index overlays on the original images.
- `pca_elements_instruct.py` – Extracts vision patch embeddings from SmolVLM on `elements` dataset, visualizing them with PCA and overlaying patch indices on the original images.

## How to run the scripts

`cd scripts`

`python filename.py`
