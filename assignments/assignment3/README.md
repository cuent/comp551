# Modified MNIST
## COMP 551 [project 3](https://cs.mcgill.ca/~wlh/comp551/files/miniproject3_spec.pdf).

In this project, we present an approach for classifying images using deep convolutional neural networks (CNNs). In particular, we analyzed $40,000$ images of hand-written digits from the Modified MNIST dataset. We demonstrate that the choice of training technique has a considerable impact on performance; specifically, we show that data augmentation combined with learning rate scheduling results in higher classification accuracies. Moreover, we demonstrate that transfer learning and ensembling improve image classification performance, achieving 99% classification accuracy on the test set.

See the results on [Kaggle](https://www.kaggle.com/c/comp-551-w2019-project-3-modified-mnist/leaderboard).


Find the code [here](https://github.com/cuent/COMP551-Project3).


## Approach
1) Use pre-trained models.
2) Use affine transformations for data augmentation (scaling, shifting, shearing).
3) Use learning rate-scheduling.
4) At the end use ensembling (i.e. voting ensemble or stacked ensemble).
