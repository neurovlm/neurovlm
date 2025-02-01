# <em>neuro</em>vlm

Neuro vision-language models.

### Model

The <em>neuro</em>vlm model is trained on text-neuro pairs and includes a neuro-autoencoder and a text encoder-aligner.
The neuro-autoencoder is first trained. The text aligner converts latent text vectors, from a pre-trained text
transformer, to the latent neuro space. This results in text-to-brain mappings.

![model](https://github.com/neurovlm/neurovlm_data/blob/08ad84c1460a4e7e46929ed5c8e89c6e462b9994/docs/model.png)

### Module

`neurovlm.models`: pytorch models

`neurovlm.train`: train models

`neurovlm.coords`: extract neuro-vectors from MNI coordinates

`neurovlm.data`: fetch dataset
  

