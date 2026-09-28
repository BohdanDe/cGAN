# conditional_gan_mnist

Conditional GAN implements the generation of handwritten digit images (from 0 to 9) based on the MNIST dataset using PyTorch. The project is educational and demonstrates the work of generative adversarial networks (GAN) with conditional generation (cGAN). 

Unlike a classic GAN, here you can pass a class label to the model (for example, the digit "3"), and the generator will create an image of exactly this digit. This is implemented through the concatenation of spatial features and nn.Embedding layers.

# Features

- The architecture is built on convolutional neural networks: Conv2d for the discriminator and ConvTranspose2d for the generator.
- Conditional integration: using nn.Embedding(10, 128) to introduce class information both into the latent noise space and into the feature maps of the discriminator.
- Label Smoothing technique is applied (Soft Labels: 0.9 for real and 0.1 for fake) to prevent discriminator overconfidence.
- Asymmetric training loop: the generator takes 3 optimization steps for every 1 step of the discriminator to stabilize training.
- Using the Adam optimizer with parameters betas=(0.5, 0.999), which is a standard for stable GANs.
- Dynamic learning rate reduction through StepLR.
- Additional scripts for visualizing the training evolution (time-lapse) and checking the generation of all classes.

# preparation

To run, you need PyTorch, Torchvision, and libraries for visualization/progress.
```bash
pip install torch torchvision matplotlib tqdm
```

## file structure

```text
project_root/
│
├── checkpoints9/ (folder where the generator weights will be saved)
├── config.py
├── main.py (main training loop)
├── models.py (network architectures)
└── plot_every_number.py (script for checking a specific checkpoint)
```
The MNIST dataset will be downloaded automatically into the ./data folder during the first run of main.py.

## run

To run the training:
```bash
python main.py
```

To view the training evolution (saved weights in the `checkpoints` folder are required):
```bash
python multimodel_plot.py
```

# neural network architecture

**Generator:**
- Input noise (vector 100) -> Linear -> ReLU
- Input label (0-9) -> Embedding(10, 128)
- Concatenation of both vectors, reshaping into spatial tensors.
- A block of ConvTranspose2d layers with a gradual increase in size (4x4 -> 8x8 -> 16x16 -> 28x28) and a decrease in the number of channels (192 -> 96 -> 48 -> 24 -> 1).
- Output activation: Tanh (images in the range [-1, 1]).

**Discriminator:**
- Input image (1x28x28) -> two blocks of Conv2d(kernel=3) + ReLU + MaxPool2d(kernel=2).
- Flatten -> Linear (extracting realism features).
- Concatenation of realism features with the class label Embedding.
- Final multilayer perceptron (MLP) to 1 output neuron. (The BCEWithLogitsLoss loss function includes Sigmoid).

# results

The training process on 25 epochs with a batch size of 1024. The training logs show the balancing between the generator and discriminator losses:

```text
 Epoch 1/25 G loss: 0.6686107001062167 D loss: 0.6944530969959194
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:11<00:00,  5.13it/s]
 Epoch 2/25 G loss: 0.6869637440826933 D loss: 0.6917215676630958
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.50it/s]
 Epoch 3/25 G loss: 0.6919154443983304 D loss: 0.6927225034115678
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.50it/s]
 Epoch 4/25 G loss: 0.6951321161399453 D loss: 0.6933205218638404
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.65it/s]
 Epoch 5/25 G loss: 0.6933246951992229 D loss: 0.6934565598681822
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.61it/s]
 Epoch 6/25 G loss: 0.6972634549868308 D loss: 0.691285387944367
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.38it/s]
 Epoch 7/25 G loss: 0.6976391955957575 D loss: 0.6901555263389976
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.58it/s]
 Epoch 8/25 G loss: 0.6979609111608085 D loss: 0.6895488712747219
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.56it/s]
 Epoch 9/25 G loss: 0.6955988982976493 D loss: 0.6900077785475779
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.65it/s]
 Epoch 10/25 G loss: 0.6999464398723537 D loss: 0.688619973295826
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.51it/s]
 Epoch 11/25 G loss: 0.7004415281748367 D loss: 0.6872831977019875
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:11<00:00,  5.17it/s]
 Epoch 12/25 G loss: 0.6972098289910009 D loss: 0.6898133663807885
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:11<00:00,  5.33it/s]
 Epoch 13/25 G loss: 0.6996363997459412 D loss: 0.6879511718022622
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.46it/s]
 Epoch 14/25 G loss: 0.7012039625038535 D loss: 0.6881368503732196
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.56it/s]
 Epoch 15/25 G loss: 0.6998234714491892 D loss: 0.6881565370802152
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.46it/s]
 Epoch 16/25 G loss: 0.6996226320832463 D loss: 0.6878586411476135
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.47it/s]
 Epoch 17/25 G loss: 0.6987899865134287 D loss: 0.6860551894721338
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:11<00:00,  5.28it/s]
 Epoch 18/25 G loss: 0.7011807318461143 D loss: 0.6874226594375352
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.52it/s]
 Epoch 19/25 G loss: 0.7027939558029175 D loss: 0.6861275070804661
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.55it/s]
 Epoch 20/25 G loss: 0.7010758983886848 D loss: 0.6877714739007464
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:10<00:00,  5.67it/s]
 Epoch 21/25 G loss: 0.7006490159842927 D loss: 0.6862726928824086
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:12<00:00,  4.65it/s]
 Epoch 22/25 G loss: 0.700922144671618 D loss: 0.6866104986708043
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:16<00:00,  3.57it/s]
 Epoch 23/25 G loss: 0.7022896984876212 D loss: 0.6857818405506975
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:14<00:00,  4.17it/s]
 Epoch 24/25 G loss: 0.7036360714395168 D loss: 0.687671838170391
----G lr: 0.0002 D lr: 2e-05
100%|██████████████████████████████████████████████████████████████████████████████████| 59/59 [00:13<00:00,  4.34it/s]
 Epoch 25/25 G loss: 0.7017703824124094 D loss: 0.689080909147101
----G lr: 0.0002 D lr: 2e-05
```

### Generation of all digits (from 0 to 9) at the 25th epoch:
![Generation of all digits](https://github.com/BohdanDe/cGAN/blob/main/Figure_2.png)

### Model training evolution
The graph shows how with every N-th epoch, the generator learns to transform noise into meaningful features and better match the given condition (digit), in this case, a three:
![Evolution of models](https://github.com/BohdanDe/cGAN/blob/main/Figure_1.png)
Although due to the relative simplicity of the model, the limited dataset, and only 25 epochs, the numbers often turn out blurry, the shapes of a three still stand out, unlike the simple noise in the first epochs
