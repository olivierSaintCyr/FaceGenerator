- Loss Handler/ log to graph previous epoch loss
- FID metric
- learning rate
_____IDEAS__________
- Transfer learning rate to train the same seed in different img sizes 
- Train multiple models scaling them up with 
- Maybe ResNet generator
- Prioritise the most efficient discriminator in training : need to dev math
- Return to initial gen model 64 filters last convolution layer ***
- Return to the fake (image) discriminator output model with dropout
- Train discriminator before genereator
_____QUESTIONS _____
# loss closer to 0 mean ???
- Understand doubles faces done
possible explaination : The discs could output images that 
- Transferer le resnet pour le GAN au lieu le reecrire de 0