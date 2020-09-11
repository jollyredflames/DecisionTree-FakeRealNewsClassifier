from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse.csr import csr_matrix
from random import randint
from sklearn import tree
import graphviz
import math

def load_data(realFilename: str, fakeFilename: str) -> \
        (CountVectorizer, csr_matrix, csr_matrix, [int], [int], csr_matrix, [int]):
    """Loads the data, preprocesses it (assigins target 1 for realFileName, and 0 for fake filename),
    and returns a tuple containing at index [i]:
    [0] CountVectorizer : can be used to transform new text into vector representation that works with this tree
    [1] csr_matrix : containing test data in sparse matrix transformed format to be used for training
    [2] [int] : containing targets for training data
    [3] csr_matrix : containing test data in sparse matrix transformed format to be used for validation
    [4] [int] : containing targets for validation data
    [5] csr_matrix : containing test data in sparse matrix transformed format to be used for testing
    [6] [int] : containing targets for test data
    """
    fileReal = open(realFilename)
    dataReal = fileReal.read().splitlines()
    fileFake = open(fakeFilename)
    dataFake = fileFake.read().splitlines()
    data = []
    target = []

    for i in range(len(dataReal)):
        data.append(dataReal[i])
        target.append(1)

    for i in range(len(dataFake)):
        data.append(dataFake[i])
        target.append(0)

    fileReal.close()
    fileFake.close()

    lenToValidate = int(len(data)*0.15)
    valSetData = []
    valSetTarget = []
    for i in range(lenToValidate):
        num = randint(0, len(data)-1)
        valSetData.append(data.pop(num))
        valSetTarget.append(target.pop(num))

    fileFake.close()
    fileReal.close()

    dataTrain, dataTest, targetTrain, targetTest = train_test_split(data, target, train_size=0.82353, shuffle=True)
    vectorizer = CountVectorizer()
    dataTrain = vectorizer.fit_transform(dataTrain)
    dataTest = vectorizer.transform(dataTest)
    valSetData = vectorizer.transform(valSetData)
    return (vectorizer, dataTrain, targetTrain, valSetData, valSetTarget, dataTest, targetTest)


def select_model(dataTrain: csr_matrix, targetTrain: [int], valSetData: csr_matrix, valSetTarget: [int]) -> DecisionTreeClassifier:
    """Returns Decision Tree Classifier with the highest accuracy on the test data set"""
    treeArr = []
    maxNum = int(len(targetTrain)**0.5)
    numsDone = []

    while len(numsDone) != 5:
        x = randint(maxNum//8, maxNum)
        if x not in numsDone:
            numsDone.append(x)
            treeArr.append(DecisionTreeClassifier(criterion="gini", max_depth=x))
            treeArr.append(DecisionTreeClassifier(criterion="entropy", max_depth=x))

    predictions = []

    for i in range(len(treeArr)):
        treeArr[i] = treeArr[i].fit(dataTrain, targetTrain)
        predictions.append(treeArr[i].predict(valSetData))

    accuracy = []

    for i in range(len(treeArr)):
        accuracy.append(getAccuracy(predictions[i], valSetTarget))
        print(accuracy)

    index = accuracy.index(max(accuracy))

    return treeArr[index]

def compute_information_gain(vectorizer: CountVectorizer, word: str, dataTrain: csr_matrix, targetTrain: [int]) \
        -> float:
    """Compute information gain of given word and return value"""
    word = word.lower()
    parentEntropy = computeEntropy(targetTrain)
    numRows = dataTrain.get_shape()[0]
    wordYesSplit = {0:0, 1:0}
    wordNoSplit = {0:0, 1:0}
    for count in range(numRows):
        simpleSentence = vectorizer.inverse_transform(dataTrain[count])[0]
        if word in simpleSentence:
            wordYesSplit[targetTrain[count]] += 1
        else:
            wordNoSplit[targetTrain[count]] += 1
    wordYesArray = wordYesSplit[0]*[0] + wordYesSplit[1]*[1]
    #print("lenYesArr: {}, YesDict: {}".format(len(wordYesArray), wordYesSplit))
    wordNoArray = wordNoSplit[0] * [0] + wordNoSplit[1] * [1]
    #print("lenNoArr: {}, NoDict: {}".format(len(wordNoArray), wordNoSplit))
    yesSplitEntropy = computeEntropy(wordYesArray)
    noSplitEntropy = computeEntropy(wordNoArray)
    probYes = len(wordYesArray) / numRows
    probNo = len(wordNoArray) / numRows
    #print("parEnt: {}, YesEnt: {}, NoEnt: {}".format(parentEntropy, yesSplitEntropy, noSplitEntropy))
    #print("probYes= {}, probNo= {}".format(probYes, probNo))

    return parentEntropy - (yesSplitEntropy*probYes + noSplitEntropy*probNo)

def runModel():
    x = load_data("clean_real.txt", "clean_fake.txt")
    y = select_model(x[1],x[2],x[3],x[4])
    createTreeImage(y, "realFakeNews")

    return (x, y)

def computeEntropy(data : [int]) -> float:
    """Compute and return entropy of binary list"""
    numZeros = data.count(0)
    #print(data)
    propZeros = numZeros / len(data)
    #print(propZeros)
    propOnes = data.count(1) / len(data)
    #print(propOnes)
    if propOnes <= 0 or propZeros <= 0:
        return 0
    return ((-1 * (propZeros)) * math.log2(propZeros)) + ((-1 * (propOnes)) * math.log2(propOnes))

def createTreeImage(classifier: DecisionTreeClassifier, filename: str) -> None:
    """output filename.pdf for this classifier"""
    raw_data = tree.export_graphviz(classifier)
    graph = graphviz.Source(raw_data)
    graph.render(filename)

def getAccuracy(prediction: [int], trueResults: [int]) -> float:
    """Returns accuracy of prediction based on true results. len(predicition) == len(trueResults)"""
    correct = 0
    wrong = 0
    for pred in range(len(prediction)):
        if prediction[pred] == trueResults[pred]:
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)
