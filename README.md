# Classification_of_Iris_Flowers
followed a tutorial to get my feet wet on making a machine learning project. 
I used a UCI dataset about Iris flowers, the dataset had 3 types of Iris flowers: Iris-setosa, Iris-Versicolor and Iris-virginica 

I first looked at the information within the dataset like the sepal_length and width (Sepal is the green part that protects the flower bud)
,and we looked at the peta_length and width as well. 
Also looked at statistical summary of the dataset like the max, min and others about each categorize that I was using 
I then made visuals for the data like a box and whisker plot, a histogram, and scatter plot matrix 
Then I split the data into 80% will be used for train and then 20% for the validation dataset
then made a list and tested out many algorithms chose the best based off of accuracy, the best one being SVM 
then made predictions on the validation dataset after we fitted the SVM on to the entire dataset.

The results were:
our model predicted Iris-Setosa perfectly
for Iris-versicolor it had perfect precision, but was a 92% for recall, suggesting that the model has false negative
for Iris-virginica it is the opposite of versicolor, with that being that recall is at 100% but precision is at 86%, suggesting that the model has false positives
The overall accuracy is 97% which is good but further changes to parameters or cleaning up the data can help the results be better. 
