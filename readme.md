
## V1
Generative Adversarial Network for 64 x 64 colored images, trained on game cards. The ensemble model consists of two competing fully connected models, a discriminator and a generator. The generator starts with a latent random noise vector, and is built up to a full image. The discriminator uses BCELoss (shown below), and tries to minimize the below. The generator on the other hand also uses BCELoss but instead tries to minimize $-\log(D(G(z)))$, in other words fool the discriminator into beleiving a generated sample is real i.e $(D(G(z)) = 1)$



<div style="display:flex;justify-content:center;align-items:center">
    <img src="./examples/V1/loss_formula.png"><img>
</div>

### Results

Below a batch of results can be seen at the begining stages of training, and after about 1000 epochs. The model is clearly training, and you can begin to see cards forming, although a clearer image is likley too complex for this model to produce. The baseline is a sample batch of training data.

![image info](./examples/V1/V1.png)

