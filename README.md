# Advserarial Image Analysis
## Todo:
    Adversarial Image Generation via Regression [x]
    Regression Training Optimizaion w/ Adversarial Image Generation [ ]
    Confidence Calculation [ ]
    Convnet Implementation [ ]
    
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
    5. Screen input thorugh both boolean masks to generate adversarial images

### Results and Analysis
An spectrum of alpha values were used to scale delta to demonstrate its effect on number of instances affect by the shift as well as the post-shift image.

![alpha500](https://image.ibb.co/iKjKyv/figure_1.png)

The respective alpha values of each figure is annotated above each figure.
For alpha=200 and 300, the adversarial iamge generator was not able to generate up to 10 adversarial images.

Serveral obvious points of interests are demonstrated in the process.
- Delta becomes more significant as alpha increases. This is intuitive and expected as delta is scaled by alpha
- The number of adversarial images generated increases as alpha increases
- Changes in the resulting adversarial images are increasingly significant as alpha increases

Despite shifting, the correct classifications for adversarial images under human eye are still clearly their original classes. Thus, it is very possible to use these examples to retrain the model in order to decrease overfitting.


