
## V1
Generative Adversarial Network for 64 x 64 colored images, trained on game cards. The ensemble model consists of two competing fully connected models, a discriminator and a generator. The generator starts with a latent random noise vector, and is built up to a full image. The discriminator uses BCELoss 

![image info](./examples/V1/loss_formula.png)

![image info](./examples/V1/V1.png)

