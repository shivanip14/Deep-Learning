# Model Card: patnet
## Model details
_patnet_ is a CNN image classification model developed by Shivani Patel, pretrained on a large dataset of museum objects, the MAMe dataset which contains thousands of images spanning 29 categories of materials.

### Model date
October 2022

### Model version
v1

### Model type
The architecture has 3 sequential blocks of conv-conv-batchnorm-maxpool, followed by 2 dense layers, and a 60% dropout layer in between. In the end it uses Adam optimiser and categorical cross-entropy for measuring loss.

The model is fairly simple, as in there are no skip connections or residual blocks. There are 291,645 total parameters, out of which 291,197 are trainable. It took roughly ~56m to train on a Tesla v100.

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
- Accuracy graphs, loss graphs and confusion matrices are also generated and saved at `./lab1data/savedmodels/accuracy/`, `./lab1data/savedmodels/loss/` and `./lab1data/savedmodels/conf/` respectively.
- The Cohen-Kappa score is used for evaluatin of classification models to compare the agreement of two or more raters - in this case, between our model and the actual test labels; however it can also be used to calculate between each pair of models to evaluate which is better.
        
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

![data_distribution](./lab1data/dataex/data_dist.png)

## Quantitative analysis
The model performance is as summarised below:

`Test accuracy = 0.7562751770019531`

`Test loss = 1.0035877227783203`

`Cohen-Kappa score = 0.7465339955958673`

                               precision    recall  f1-score    support
             Oil on canvas       0.96       0.95      0.95       700
                  Graphite       0.77       0.71      0.74       700
                     Glass       0.77       0.75      0.76       700
                 Limestone       0.78       0.80      0.79       313
                    Bronze       0.69       0.74      0.72       700
                   Ceramic       0.74       0.60      0.66       700
          Polychromed wood       0.73       0.79      0.76       700
                   Faience       0.61       0.71      0.66       700
                      Wood       0.87       0.87      0.87       700
                      Gold       0.57       0.98      0.72       188
                    Marble       0.95       0.96      0.95       328
                     Ivory       0.97       0.93      0.95       584
                    Silver       0.73       0.76      0.75       265
                   Etching       0.72       0.60      0.65       572
                      Iron       0.79       0.55      0.65       700
                 Engraving       0.80       0.70      0.75       700
                     Steel       0.38       0.75      0.51       257
                 Woodblock       0.81       0.79      0.80       700
     Silk and metal thread       0.63       0.90      0.74       286
                Lithograph       0.55       0.57      0.56       375
              Woven fabric       0.77       0.85      0.81       700
                 Porcelain       0.32       0.65      0.43        95
         Pen and brown ink       0.82       0.71      0.76       700
                   Woodcut       0.66       0.83      0.74       133
            Wood engraving       0.65       0.76      0.70       700
    Hand-colored engraving       0.68       0.83      0.75       361
                      Clay       0.92       0.85      0.88       700
      Hand-colored etching       0.95       0.80      0.87       700
        Albumen photograph       0.87       0.54      0.66       700

                  accuracy                            0.76     15657
                 macro avg       0.74       0.77      0.74     15657
              weighted avg       0.78       0.76      0.76     15657

As is evident, the scores across some of the categories is a little subpar as compared to the average performance of the model as a whole, namely Steel, Porcelain & Lithograph. The latter two can be attributed to less no. of test instances, however in case of Steel, even if the number is as high as other categories performing good, it could be a model bias that is failing to generalise Steel objects.

On the other hand, there are some which are performing exceptionally well, namely Oil on canvas, Marble & Ivory. 

## Ethical considerations
- The model is trained on data that is publicly available, on commercial websites and/or as pre-packaged datasets used widely by the AI community. Most (if not all) art objects photographed are on public display at various museums across the US.
- The data is about museum art and artifcats and does not contain any sensitive or PII (personally identifyable information).

## Caveats and recommendations
- Models with similar accuracy/val accuracy can be compared to each other with the Kappa score to see which ones are actually better - in-essence a Cohen-Kappa matrix for each of these similar models.
- The impact of data augmentation have not been able to be fully explored, mainly due to very long training times, especially as the original dataset is big enough. Nevertheless, this should be one sub-domain to look into while trying to better the model.
- Further design changes and tweaks are required for the subsequent versions of the model to reach the threshold of 80% test accuracy, after which work on the variable-sized images can be started. There is an evident scarcity of solutions to make use of the variable shaped-input in a single model and still provide real-life deployment-level performance, and much of the experimentation lies there.

