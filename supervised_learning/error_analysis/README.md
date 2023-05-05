# Error Analysis 

> Error Analysis refers to the process of examining dev set examples that your algorithm misclassified, so that we can understand the underlying causes of the errors. This can help us prioritize on which problem deserves attention and how much. It gives us a direction for handling the errors.

At the end of this project I was able to solve these conceptual questions:

* What is the confusion matrix?
* What is type I error? type II?
* What is sensitivity? specificity? precision? recall?
* What is an F1 score?
* What is bias? variance?
* What is irreducible error?
* What is Bayes error?
* How can you approximate Bayes error?
* How to calculate bias and variance
* How to create a confusion matrix

## Tasks :heavy_check_mark:

0. Function def create_confusion_matrix(labels, logits): that creates a confusion matrix
1. Function def sensitivity(confusion): that calculates the sensitivity for each class in a confusion matrix
2. Function def precision(confusion): that calculates the precision for each class in a confusion matrix
3. Function def specificity(confusion): that calculates the specificity for each class in a confusion matrix
4. Function def f1_score(confusion): that calculates the F1 score of a confusion matrix.
5. Lettered answer to the question of how you should approach the following scenarios
      - High Bias, High Variance
      - High Bias, Low Variance
      - Low Bias, High Variance
      - Low Bias, Low Variance
6. Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is

## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-create_confusion.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/error_analysis/0-create_confusion.py)|
| [1-sensitivity.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/error_analysis/1-sensitivity.py)|
| [2-precision.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/error_analysis/2-precision.py)|
| [3-specificity.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/error_analysis/3-specificity.py)|
| [4-f1_score.py](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/error_analysis/4-f1_score.py)|
| [5-error_handling](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/error_analysis/5-error_handling)|
| [6-compare_and_contrast](https://github.com/otalorajuand/holbertonschool-machine_learning/blob/main/supervised_learning/error_analysis/6-compare_and_contrast)|


### Try It On Your Machine :computer:
```bash
git clone https://github.com/otalorajuand/holbertonschool-machine_learning.git
cd error_analysis
./main_files/MAINFILE.py
```
