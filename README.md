# <em>neuro</em>vlm

Neuro vision-language models.

### Model

The <em>neuro</em>vlm model is trained on text-neuro pairs and includes a neuro-autoencoder and a text encoder-aligner.
The neuro-autoencoder is first trained. The text aligner converts latent text vectors, from a pre-trained text
transformer, to the latent neuro space. This results in text-to-brain mappings.

![model](https://github.com/neurovlm/neurovlm_data/blob/c98329702fae18cfda711554c2427729a7cad496/docs/model.png).

### Module

`neurovlm.models`: pytorch models

`neurovlm.train`: train models

`neurovlm.coords`: extract neuro-vectors from MNI coordinates

`neurovlm.data`: fetch dataset
  

