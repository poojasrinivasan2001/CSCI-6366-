###  Neural Narrators: A Comparative Study of CNN-GRU and VAE-GRU for Image Captioning

This project implements and compares a baseline CNN+GRU image captioning model against four advanced Variational Autoencoder (VAE) + GRU models, each incorporating different attention mechanisms. The goal is to evaluate how attention and latent feature modeling affect the quality of generated captions.

---

#### Repository Structure

#### 1. MODEL1_NN.py
- **Description**: Baseline model using InceptionV3 (CNN) for feature extraction and a GRU decoder with Bahdanau (additive) attention.
- **Training Dataset**: Flickr8k (8,000+ images, 5 captions per image)
- **Key Features**: 
  - Image preprocessing and feature extraction with InceptionV3
  - Tokenizer-based text preprocessing
  - Attention-based GRU caption generation

---

#### 2. VAE_WithoutAttention.ipynb
- **Description**: A Variational Autoencoder (VAE) is used to encode images into a latent vector `z`, followed by a GRU decoder without any attention.
- **Training Dataset**: Flickr1k (1,000 images, 5 captions per image)
- **Key Features**:
  - VAE-style latent encoding
  - Caption generation using sampled latent space
  - No attention mechanism

---

#### 3. VAE_Additive_Attention_final.ipynb
- **Description**: Builds on the VAE baseline by integrating Bahdanau (additive) attention in the GRU decoder.
- **Training Dataset**: Flickr1k (1,000 images, 5 captions per image)
- **Key Features**:
  - VAE latent space
  - Additive attention to guide captioning
 

---

#### 4. VAE_Cross_Attention_.ipynb
- **Description**: Introduces a cross-attention mechanism where the latent representation interacts with embedded decoder inputs.
- **Training Dataset**: Flickr1k (1,000 images, 5 captions per image)
- **Key Features**:
  - VAE encoder + GRU decoder
  - Cross-attention between image and text modalities
  

---

#### 5. VAE_Self_Attention.ipynb
- **Description**: Implements a VAE encoder with a self-attention-based decoder to model intra-caption dependencies.
- **Training Dataset**: Flickr1k (1,000 images, 5 captions per image)
- **Key Features**:
  - Self-attention in decoder
  
  

---

#### 📊 Datasets

| Model File                     | Dataset   | Download Link |
|-------------------------------|-----------|----------------|
| MODEL1_NN.py                | Flickr8k  | [Flickr8k Dataset](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k) |
| All VAE Models (*.ipynb)    | Flickr1k  | [Flickr1k on Kaggle](https://www.kaggle.com/datasets/keenwarrior/small-flicker-data-for-image-captioning) |

---

#### Evaluation

Captions are evaluated using **GPT-4 based semantic scoring** instead of traditional metrics like BLEU or CIDEr. Each generated caption is assessed for:
- **Fluency**
- **Relevance**

Evaluation is performed on a **5-point ordinal scale**, where 1 is poor and 5 is excellent.

---

#### Requirements

To run this project, install the following Python packages:

```bash
pip install tensorflow numpy pandas matplotlib nltk scikit-learn tqdm
