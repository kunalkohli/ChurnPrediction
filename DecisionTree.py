import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

#store m_depth list with the values for the depth specified
m_depth = [0,2,4,8,16]						#0 in this list resembles None ie default parameter for depth
m_leaf = [2,4,8,16,32,64,128,256]			#store lsit of the number of leaf nodes specified

####################################################################################################
########reading file to determine the number of rows and columns in the file for initiating the vectors#########
file = open("training_set.csv", "r") 
read = file.readlines()		
number_of_rows = len(read)
fields = read[0].split(',')
number_of_columns = len(fields)
####################################################################################################

#initiate a  dataset with the number of rows and one less column number in the csv file 
#since id in the csv file is not needed for analysis we create the list of one column less than dimension of csv fil

dataset = np.zeros((number_of_rows,number_of_columns - 1)) #this has both the inputs and the target values in it
target = np.zeros(number_of_rows)						#initiate a target with one colums and 4661 rows

countEarlyLate = [0,0] 						#store count of early adopter(index = 0) and late adopters(index = 1) in a list

#define list of training vectors and test vector for input data and target data
training_input = list()
training_target = list()
testing_input = list()
testing_target = list()

#define list of pruned training vectors and test vector for input data and target data
training_pruned_input = list()
training_pruned_target = list()
testing_pruned_input = list()
testing_pruned_target = list()

def readFileAndFormArray(countEarlyLate): 		#convert csv file to a create a dataset
	file = open("training_set.csv", "r") 
	read = file.readlines()						#read all the dataset lines
	for i in range(0,len(read)):				#go through each line in the dataset
		processText(read[i],i,countEarlyLate)	#function to do manual encoding of each data read and append to the dataset
		target[i] = dataset[i,8]				#append values of the dataset at position(8) to the target ie adoption status
	
def processText(data,row_index,countEarlyLate): #used for encoding string values to numeric values
	data = data.strip()
	fields = data.split(',')	
	for i in range(1,len(fields)):
		#Gender
		if fields[i] == "F":
			dataset[row_index][i-1]  = 1
		elif fields[i] == "M":
			dataset[row_index][i-1]  = 0

		#Marital Status
		if fields[i] == "Single":
			dataset[row_index][i-1]  = 0
		elif fields[i] == "Married":
			dataset[row_index][i-1]  = 1
		

		#Type of Usage
		if fields[i] == "Low":
			dataset[row_index][i-1]  = 0
		elif fields[i] == "Medium":
			dataset[row_index][i-1]  = 1
		elif fields[i] == "Heavy":
			dataset[row_index][i-1]  = 2
		elif fields[i] == "PrePaid":
			dataset[row_index][i-1]  = 3
	
		#Automatic/Non-Automatic
		if fields[i] == "Automatic":
			dataset[row_index][i-1]  = 0
		elif fields[i] == "Non-Automatic":
			dataset[row_index][i-1]  = 1

		#Contract/No-Contract
		if fields[i] == "No Contract":
			dataset[row_index][i-1]  = 0
		elif fields[i] == "12 Months":
			dataset[row_index][i-1]  = 1
		elif fields[i] == "24 months":
			dataset[row_index][i-1]  = 2
		elif fields[i] == "36 Months":
			dataset[row_index][i-1]  = 3

		#Yes/No Encoding
		if fields[i] == "N":
			dataset[row_index][i-1]  = 0
		elif fields[i] == "Y":
			dataset[row_index][i-1]  = 1

		#add values of target to the data set too create a dataset with all the elements in the csv file
		if (fields[i] == "Early" or fields[i] == "Very Early"):
			dataset[row_index][i-1] = 1
			countEarlyLate[0] += 1
		elif (fields[i] == "Late" or fields[i] == "Very Late"):
			dataset[row_index][i-1]  = 2
			countEarlyLate[1] += 1			

		#Age
		dataset[row_index][1]  = fields[2] #after analysing the third column is the age ie index 2

def formVectors(training_input,training_target,testing_input,testing_target):	#create numpy training arrays for analysis
		
	for i in range(0,dataset.shape[0]):
		new_row = dataset[i][0:8] # read from 0 to 7 as to avoid reading target value
		if (i % 10 == 0):
			testing_input.append(new_row)
			testing_target.append(target[i])
		else:
			training_input.append(new_row)
			training_target.append(target[i])

	#Convert list to numpy arrays for analysis
	training_input = np.array(training_input)
	training_target = np.array(training_target)
	testing_input = np.array(testing_input)
	testing_target = np.array(testing_target)

#create pruned vectors from the complete data set created initially in processText function which had all inputs and targets
def formPrunedVectors(prunedDataset,prunedTarget,training_pruned_input,training_pruned_target,testing_pruned_input,testing_pruned_target):

	#countEarlyLate stores the count of different target values in it.since two value targets hence two values of count in the list

	prunedArraySize = 2 * min(countEarlyLate) #size of the pruned array to be twice than that of count of target which occurs less
	if (countEarlyLate[1] < countEarlyLate[0]): # if second target value count is less than first target value,sort in reverse order
		prunedtempdataset = sorted(dataset,key=lambda x:x[8],reverse = True) #sort based on last column ie adopter status to divide dataset
	else:
		prunedtempdataset = sorted(dataset,key=lambda x:x[8]) #sort based on last column ie adopter status to divide dataset
	prunedtempdataset = np.array(prunedtempdataset)
	np.random.shuffle(prunedtempdataset)					  #randomise the newly created pruned dataset
	prunedDataset = prunedtempdataset[0:prunedArraySize,0:8] #input dataset - rows till twice pruned size and columns = 8 ie only inputs
	prunedTarget = prunedtempdataset[0:prunedArraySize,8]	#create a pruned target with the last column of the pruned dataset

	#divide pruned dataset into training and testing data set in 9:1 ratio
	for i in range(0,prunedDataset.shape[0]):
		new_row = prunedDataset[i][:] 
		if (i % 10 == 0):
			testing_pruned_input.append(new_row)
			testing_pruned_target.append(target[i])
		else:
			training_pruned_input.append(new_row)
			training_pruned_target.append(target[i])
	#convert newly created lists into numpy arrays for easy analysis later
	training_pruned_input = np.array(training_pruned_input)
	training_pruned_target = np.array(training_pruned_target)
	testing_pruned_input = np.array(testing_pruned_input)
	testing_pruned_target = np.array(testing_pruned_target)

#create tree based on the m_leaf and m_depth in the list created earlier
def formTree(trainData,trainTarget,testData,testTarget,modelname):
	for i in range(0,len(m_depth)):
		accuracyList = list()			#create an accuracy list to store accuracy values for a m_depth for different m_leaf_nodes
		#create tree with different m_leaf nodes specified for a particular depth 
		for j in range(0,len(m_leaf)):
			#if depth = 0 in the array ie None value hence create tree with default value of max_depth
			if (m_depth[i] == 0):
				clf = tree.DecisionTreeClassifier(max_leaf_nodes = m_leaf[j])
				clf.fit(trainData, trainTarget)
			else:
				clf = tree.DecisionTreeClassifier(max_depth=m_depth[i],max_leaf_nodes = m_leaf[j])
				clf.fit(trainData, trainTarget)
			
			#initialise count of correct and incorrect from zero for a each leaf node value 
			correct = 0
			incorrect = 0
			#create prediction of the test data set based on the tree model created
			predictions = clf.predict(testData)
	
			#test the accuracy of the predicted data from tree with the test target
			for index in range(0, predictions.shape[0]):	
				if (predictions[index] == testTarget[index]):
					correct += 1
				else:
					incorrect += 1
			#append  accuracy for a particular max_lead to the accuracy list
			accuracyList.append(float(correct)/(correct+incorrect))
		
		#plot accuracy values of for different leaf nodes for a particular depth
		plt.plot(m_leaf,accuracyList,color='red')
		plt.xticks(range(0,m_leaf[7],25))
		plt.xlabel("Number of Leaf nodes")
		plt.ylabel("Accuracy")
		#if depth = 0 then title should be None.
		if(m_depth[i] == 0):
			plt.title("Depth : None")
		else:
			plt.title(" %s , Depth : %.1f " % (modelname,float(m_depth[i])))
		plt.suptitle("Accuracy Curves for different max depths and Max leaf nodes",color = 'blue')		
		plt.show()


#######################################
#create data set for nomal and pruned dataset

readFileAndFormArray(countEarlyLate)	
formVectors(training_input,training_target,testing_input,testing_target)

prunedDataset = np.zeros((2*min(countEarlyLate),8))
prunedTarget = np.zeros(min(countEarlyLate))

formPrunedVectors(prunedDataset,prunedTarget,training_pruned_input,training_pruned_target,testing_pruned_input,testing_pruned_target)

#forming tree prediction for all inputs

formTree(training_input,training_target,testing_input,testing_target,'Normal Model')

#forming tree prediction for pruned inputs

formTree(training_pruned_input,training_pruned_target,testing_pruned_input,testing_pruned_target,'Pruned Model')
#####################














