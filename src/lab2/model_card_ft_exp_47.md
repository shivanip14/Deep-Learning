# Model Card: MAMe-FT-DenseNet121
## Model details
_MAMe-FT-DenseNet121_ (FT experiment #47) is a CNN image classification model developed by Shivani Patel, which is essentially a DenseNet121 pre-trained on imagenet and fine-tuned for the MAMe dataset, which contains thousands of images of museum objects spanning 29 categories of materials.

### Model date
November 2022

### Model version
v1

### Model type
Originally proposed by [Gao Hunag et al in 2016](https://arxiv.org/abs/1608.06993), Dense Convolutional Network (DenseNet) has connections between each layer to every other layer in a feed-forward fashion. As compared to traditional convolutional networks with L layers having L connections (one between each layer and its subsequent layer), DensetNet has L(L+1)/2 direct connections.
Such short connections help alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters as compared to similarly deep traditional CNNs.

The architecture has 120 conv layers across 5 "dense blocks". We use the model from the Keras library, without the top classifying layer, flatten the output and instead add our own dense layer with softmax activation for classification with 29 neurons.
In the end it is compiled with SGD optimiser with LR=0.001 and categorical cross-entropy for measuring loss.

There are 8,938,077 total parameters, out of which 1,900,573 are trainable as we had frozen all the original DenseNet121 layers (keeping only the last custom added layer as trainable). It took roughly ~44m to train on a Tesla v100.
![ft_densetnet_arch](MAMe-FT-DenseNet121_arch.png)

### License
The code is licensed under BSD 3-Clause License that requires adding the developer's permission before mentioning their name for any work built on top of the software. More restrictive than MIT, less so than GPL.

### Further info
Any further questions or comments about the model should be emailed to `shivani.patel@estudiantat.upc.edu` 

## Intended use
### Primary intended uses
The model is intended for use only within the academic realm. It is built for the partial fulfillment of the course of Deep Learning in the Masters degree of Artificial Intelligence at UPC, Barcelona. However, since the MAMe classification task is an open and unsolved challenge as of the time of completion of this model, with the appropriate credit/permission, it can also be published (as is or as a base model along with necessary improvements) with a paper towards a broader use.

### Primary intended users
Academicians, students and researchers in the field of deep learning, AI or likewise, as well as hobbyists who would like to explore the domain of CNNs to solve the MAMe task.

### Out-of-scope use-cases
Any deployed use case of the model - whether commercial or not - is currently out of scope. Non-deployed uses even in a constrained environment, for any dataset other than what the model has been tested on (MAMe 256) is not advisable as thorough safety and compatibility testing have not been performed yet.

## Factors
- The classification of training, validation and test data across 29 mediums (i.e. materials and techniques) have been supervised by art experts.
- Further collaboration details can be found on the [website](https://hpai.bsc.es/MAMe-dataset/) and [Kaggle page](https://www.kaggle.com/datasets/ferranpares/mame-dataset).


## Metrics
The metrics used to evaluate the models built as part of the lab task are: accuracy, loss & Kappa score (model-wide), precision, recall & F1 score (across each category, per model).

- Good models are selected first on highest validation accuracy, and are chosen for further builds iteratively.
- Accuracy graphs, loss graphs and confusion matrices are also generated and saved at `./lab2data/savedmodels/accuracy/`, `./lab2data/savedmodels/loss/` and `./lab2data/savedmodels/conf/` respectively.
- The Cohen-Kappa score is used for evaluation of classification models to compare the agreement of two or more raters - in this case, between our model and the actual test labels; however it can also be used to calculate between each pair of models to evaluate which is better.
        
        Kappa score       Agreement
        <0                Less than just a chance agreement (disagreement, even)
        0.01-0.20   	    Slight agreement
        0.21-0.40   	    Fair agreement
        0.41-0.60   	    Moderate agreement
        0.61-0.80   	    Substantial agreement
        0.81-0.99   	    Almost perfect agreement
        
- After a model is built & trained, it is evaluated using the `calculate_scores.py` which loads the model from a previously-saved JSON & weights, calculates the latter of the metrics specified above & generates a report for the model/experiment specified. (submitting a slurm job by passing experiment number as argument: e.g. `sbatch calculate_scores_launcher.sh exp_14`)

## Data
The MAMe dataset is a novel image classification task focused on museum art mediums. Originally introduced in [this](https://arxiv.org/abs/2007.13693) 2020 paper as an image classification dataset with remarkable high resolution and variable shape properties, it's goal is to provide a tool for studying the impact of such properties in image classification.
Images of thousands of artworks and artifacts from all the 3 museums (Metropolitan Museum of Art of New York, Los Angeles County Museum of Art, and Cleveland Museum of Art) is aggregated by art experts into 29 classes of mediums (i.e., materials and techniques).
While the original, high res variable sized images are not used in this model since it needs additional advanced experimentation to reach the required baseline performance of 80% test accuracy, the 256 version was the one on which the data was trained, validated and tested.

Subsets for train (20,300 instances), validate (1,450 instances) and test (15,657 instances) have been provided in the metadata `MAMe_dataset.csv`, and have a fair disribution across all the categories:

![data_distribution](../lab1/lab1data/dataex/data_dist.png)

## Quantitative analysis
The model performance is as summarised below. It was able to beat the _patnet_ model from Lab 1 which was built and trained from scratch, in case of validation accuracy by a slight margin (0.7407 in patnet vs. 0.7476 in fine-tuned), however the performance on the test set was slightly worse (test accuracy 0.7563 in patnet vs. 0.7488 in fine-tuned, Kappa score 0.7465 in patnet vs. 0.7387 in fine-tuned).

    Test accuracy = 0.7488024234771729
    Test loss = 10.459491729736328
    Cohen-Kappa score = 0.7387465006650811
    
                            precision    recall  f1-score   support

             Oil on canvas       0.94      0.93      0.94       700
                  Graphite       0.71      0.63      0.67       700
                     Glass       0.77      0.77      0.77       700
                 Limestone       0.80      0.84      0.82       313
                    Bronze       0.70      0.70      0.70       700
                   Ceramic       0.68      0.57      0.62       700
          Polychromed wood       0.78      0.76      0.77       700
                   Faience       0.72      0.70      0.71       700
                      Wood       0.83      0.82      0.82       700
                      Gold       0.65      0.79      0.71       188
                    Marble       0.91      0.97      0.94       328
                     Ivory       0.91      0.95      0.93       584
                    Silver       0.58      0.75      0.65       265
                   Etching       0.66      0.63      0.64       572
                      Iron       0.76      0.66      0.70       700
                 Engraving       0.72      0.67      0.69       700
                     Steel       0.44      0.77      0.56       257
                 Woodblock       0.81      0.72      0.77       700
     Silk and metal thread       0.70      0.81      0.75       286
                Lithograph       0.53      0.64      0.58       375
              Woven fabric       0.90      0.84      0.87       700
                 Porcelain       0.29      0.57      0.39        95
         Pen and brown ink       0.72      0.71      0.71       700
                   Woodcut       0.63      0.91      0.74       133
            Wood engraving       0.73      0.64      0.68       700
    Hand-colored engraving       0.59      0.75      0.66       361
                      Clay       0.88      0.88      0.88       700
      Hand-colored etching       0.82      0.81      0.82       700
        Albumen photograph       0.84      0.73      0.78       700

                  accuracy                           0.75     15657
                 macro avg       0.72      0.76      0.73     15657
              weighted avg       0.76      0.75      0.75     15657


As is evident, the scores across some of the categories is a little subpar as compared to the average performance of the model as a whole, especially `Porcelain`.

Categories like `Steel` were performing quite bad on with a model developed from scratch (i.e. _patnet_ from Lab 1), however with a fine-tuned model such as the current one, we see slight improvement (from 0.51 to 0.56).

On the other hand, there are some which are performing exceptionally well, namely `Oil on canvas`, `Marble` & `Ivory`. 

The resultant plots for the same are as below:

Accuracy | Loss
:--------:|:-------:
![acc_exp47](./lab2data/savedmodels/accuracy/ft_47.png) | ![loss_exp47](./lab2data/savedmodels/loss/ft_47.png)
![conf_exp47](./lab2data/savedmodels/conf/47_confusion_matrix.png)

## Ethical considerations
- The model is trained on data that is publicly available, on commercial websites and/or as pre-packaged datasets used widely by the AI community. Most (if not all) art objects photographed are on public display at various museums across the US.
- The data is about museum art and artifcats and does not contain any sensitive or PII (personally identifiable information).

## Caveats and recommendations - TODO
- Models with similar accuracy/val accuracy can be compared to each other with the Kappa score to see which ones are actually better - in-essence a Cohen-Kappa matrix for each of these similar models.
- Further design changes and tweaks are required for the subsequent versions of the model to reach the threshold of 80% test accuracy, after which work on the variable-sized images can be started. There is an evident scarcity of solutions to make use of the variable shaped-input in a single model and still provide real-life deployment-level performance, and much of the experimentation lies there.

