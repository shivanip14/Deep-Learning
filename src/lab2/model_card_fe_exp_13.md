# Model Card: MAMe-FE-SVMX
## Model details
_MAMe-FE-SVMX_ is an SVM image classification model developed by Shivani Patel, which is essentially a multi-class classifier trained on features extracted from an Xception network (pre-trained on imagenet).
All the conv layers and both dense layers from the end from the underlying feature extractor CNN are chosen and a full network embedding ([D. Garcia-Gasulla et al, 2017](https://arxiv.org/abs/1705.07706)) is generated from the training data, and the resultant features are used in training the said SVM classifier.

### Model date
November 2022

### Model version
v1

### Model type
Originally proposed by [François Chollet in 2016](https://arxiv.org/abs/1610.02357), Xception (or an eXtreme version of Inception) features modified depth-wise separable convolutions, where a pointwise convolution is followed by a depthwise convolution (instead of the other way around in case of Inception network).
The Xception architecture has the same number of parameters as InceptionV3, hence the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.
In the end it is compiled with SGD optimiser with LR=0.001 and categorical cross-entropy for measuring loss.

![arch](MAMe-FA-SVMX_arch.png)

We used a total of 42 layers (all 40 conv layers from the Xception layer + 2 FC layers in the end) for the feature extraction from the training data. These were then fed to a Linear SVC for training with default out-of-the-box parameters (L2 penalty, squared hinge loss, tolerance of 0.0001, C=1.0 and no class weighing).


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
- Accuracy graphs, loss graphs and confusion matrices are also generated and saved at `./lab2data/savedmodels/fe/accuracy/`, `./lab2data/savedmodels/fe/loss/` and `./lab2data/savedmodels/fe/conf/` respectively.
- The Cohen-Kappa score is used for evaluation of classification models to compare the agreement of two or more raters - in this case, between our model and the actual test labels; however it can also be used to calculate between each pair of models to evaluate which is better.
        
        Kappa score       Agreement
        <0                Less than just a chance agreement (disagreement, even)
        0.01-0.20   	    Slight agreement
        0.21-0.40   	    Fair agreement
        0.41-0.60   	    Moderate agreement
        0.61-0.80   	    Substantial agreement
        0.81-0.99   	    Almost perfect agreement   <-- this model falls here
        

## Data
The MAMe dataset is a novel image classification task focused on museum art mediums. Originally introduced in [this](https://arxiv.org/abs/2007.13693) 2020 paper as an image classification dataset with remarkable high resolution and variable shape properties, it's goal is to provide a tool for studying the impact of such properties in image classification.
Images of thousands of artworks and artifacts from all the 3 museums (Metropolitan Museum of Art of New York, Los Angeles County Museum of Art, and Cleveland Museum of Art) is aggregated by art experts into 29 classes of mediums (i.e., materials and techniques).
While the original, high res variable sized images are not used in this model since it needs additional advanced experimentation to reach the required baseline performance of 80% test accuracy, the 256 version was the one on which the data was trained, validated and tested.

Subsets for train (20,300 instances), validate (1,450 instances) and test (15,657 instances) have been provided in the metadata `MAMe_dataset.csv`, and have a fair disribution across all the categories:

![data_distribution](../lab1/lab1data/dataex/data_dist.png)

## Quantitative analysis
The model performance is as summarised below. It was able to easily beat the _patnet_ model from Lab 1 which was built and trained from scratch, in case of validation accuracy by more than 10% (0.7407 in patnet vs. 0.8455 in SVM/FE). The performance on the test set was also significantly better (test accuracy 0.7563 in patnet vs. 0.8500 in SVM/FE, Kappa score 0.7465 in patnet vs. 0.8440 in SVM/FE).

    Test accuracy = 0.8500351280577377
    Cohen-Kappa score = 0.8439702424033271
                            precision    recall  f1-score   support
    
        Albumen photograph       0.98      0.98      0.98       700
                    Bronze       0.81      0.81      0.81       700
                   Ceramic       0.85      0.83      0.84       700
                      Clay       0.89      0.92      0.90       313
                 Engraving       0.83      0.80      0.81       700
                   Etching       0.80      0.75      0.77       700
                   Faience       0.89      0.83      0.86       700
                     Glass       0.87      0.84      0.86       700
                      Gold       0.89      0.92      0.90       700
                  Graphite       0.81      0.95      0.87       188
    Hand-colored engraving       0.97      0.99      0.98       328
      Hand-colored etching       0.96      0.98      0.97       584
                      Iron       0.74      0.85      0.79       265
                     Ivory       0.79      0.79      0.79       572
                 Limestone       0.85      0.76      0.80       700
                Lithograph       0.88      0.80      0.84       700
                    Marble       0.60      0.82      0.70       257
             Oil on canvas       0.84      0.82      0.83       700
         Pen and brown ink       0.83      0.95      0.89       286
          Polychromed wood       0.64      0.67      0.66       375
                 Porcelain       0.89      0.94      0.91       700
     Silk and metal thread       0.32      0.77      0.46        95
                    Silver       0.91      0.82      0.86       700
                     Steel       0.76      0.95      0.85       133
                      Wood       0.82      0.76      0.79       700
            Wood engraving       0.77      0.89      0.82       361
                 Woodblock       0.95      0.97      0.96       700
                   Woodcut       0.91      0.90      0.91       700
             Woven fabric        0.92      0.76      0.84       700
    
                  accuracy                           0.85     15657
                 macro avg       0.83      0.86      0.84     15657
              weighted avg       0.86      0.85      0.85     15657

As is evident, the scores across some of the categories is a little subpar as compared to the average performance of the model as a whole, especially `Silk and metal thread`.

Particularly hard categories like `Steel`, `Porcelain` & `Lithograph` were performing quite bad on traditional CNN models (either ones developed from scratch like _patnet_ from Lab 1 or fine-tuned), surprisingly surpassed expectatons with feature extraction on an imagenet-pre-trained model such as the current one, with a huge improvement in results (F1 scores improved from 0.51 to 0.85, from 0.43 to 0.91, from 0.56 to 0.84 respectively). 

## Ethical considerations
- The model is trained on data that is publicly available, on commercial websites and/or as pre-packaged datasets used widely by the AI community. Most (if not all) art objects photographed are on public display at various museums across the US.
- The data is about museum art and artifcats and does not contain any sensitive or PII (personally identifiable information).

## Caveats and recommendations
- Some classifiers like AdaBoost, Decision Trees and Gaussian Naive-Bayes when trained on the _**same set of extracted features**_, proved to be quite subpar. Perhaps some sort of weighing function could be used to weigh in/aggregate the performance across classifiers to provide a more comprehensive score for a single model.
- Owing to huge variation in performance for the same set of extracted training features across diff categories of classifiers, it is proved that proper choice of the classifier is as important as selecting which layers to extract the activations from.
- For the set of extracted features, SVM and Random Forest both worked well but SVM was still more than 10% better performing than RFs.
- Since it crossed the threshold of 80% for the fixed sixed dataset, the same model could also be tweaked for further work on the variable-sized images.

