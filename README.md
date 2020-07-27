# Analyzing Self Supervised and Supervised Learning Methods on STL-10 Dataset
In this paper, we described various methods to perform classification task on STL-10 dataset. The target of this work was
two-fold: first, trying out various self supervised approaches, analyzing and comparing the feature representation for the
task in hand and second trying out a novel approach named ”Harmonic Networks” with our enhancements to come up with a highly 
efficient approach for classifying STL-10 data which doesn’t use the unlabeled data for training. We performed popular 
pretext tasks including Rotation Prediction, JigSaw Puzzle Solving, Inpainting under same experimental conditions and 
downstream architecture and analyzed the essence of features representation learnt during pretext phase for downstream 
classification task. We found that among all these techniques Jigsaw Puzzle as pretext task provides the most suitable feature
learning for classification. Also we found that the pretext models with very high training accuracy perform worse than the 
models with relatively low training accuracy, which goes on to show that they overfit the data and the features learned are 
far less meaningful. Finally, we trained Harmonic network and got a classification accuracy of 90.56%.
