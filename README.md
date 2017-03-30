# Advserarial Image Analysis
## Todo:
    Adversarial Image Generation via Regression [x]
    Convnet Implementation [X]
    Training Optimizaion w/ Adversarial Image Generation [ ]
    
## Logistic Regression Analysis
### Introdution and Methods
Adversarial generation is implemented on a simple logistic regression model via Tensorflow and the MNIST dataset.
The generating process is conducted in a directed manner, targetting changing input labeleld '2' to be classified as '6'. 
The generation process is as follows:
    1. Train logistic regression model and save weights
    2. Generate boolean mask for correctly classified instances of label '2'
    3. Calculate delta by as (alpha) * (gradient of modified cost function over input)
    3. Shift input vector towards the label '6' by delta
    4. Generate boolean mask for classified instances of label '6' post shift of input
    5. Screen concatenated input (input,delta,postshift_input) both boolean masks to generate adversarial images

### Results
An spectrum of alpha values were used to scale delta to demonstrate its effect on number of instances affect by the shift as well as the post-shift image.

![alpha500](https://image.ibb.co/iKjKyv/figure_1.png)

The respective alpha values of each figure is annotated above each figure.
For alpha=200 and 300, the adversarial iamge generator was not able to generate up to 10 adversarial images.

Serveral obvious points of interests are demonstrated in the process.
- Delta becomes more significant as alpha increases. This is intuitive and expected as delta is scaled by alpha
- The number of adversarial images generated increases as alpha increases
- Changes in the resulting adversarial images are increasingly significant as alpha increases
- Overall images lost "blackness" as alpha increased. Due to the increased absolute value of negative delta values which sets the lower limit for blackness. 

Despite shifting, the correct classifications for adversarial images under human eye are still clearly their original classes. Thus, it is very possible to use these examples to retrain the model in order to decrease overfitting.

## ConvNet Analysis
### Methods and Results
Methods are largely the same as for the regression model. However trained, optimized, and delta calculated with a ConvNet model. 
Similary to the generating process used for regression, an array of alpha values were used to scale the gradient. 

![alpha500](:http://imgur.com/f5127c9d-403e-498e-8772-3658ca5a14bf)

Several interesting pieces of result came to attention:
- Convnet required a higher alpha value to scale the delta in order to generate 10 adversarial images. Alpha at 200 and 300 were unable to generate any adversarial examples. 
- By eye, the delta images are significantly more obvious. When comparing alpha=700 from the ConvNet figure and alpha=300 from the Regression figure, the ConvNet generating process was only able to produce 4 examples compared to 5 by Regression. Yet, the delta values for the ConvNet model is much more significant. In other words, the ConvNet model required a higher delta to generate even less adversarial images. 
- The "blackness" of the figures degraded much more for the ConvNet model. Which means that the gradient is more extreme. 


