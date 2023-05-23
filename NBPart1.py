# NaiveBayes.py
# Learns to predict labels using probabilities
# Aviva Blonder

import random
import math
import sys

class NaiveBayes:
    
    attributes = [] # List of attributes
    trainsize = 0 # number of instances in the trainset
    pLabel = {} # Dictionary of labels and their probabilities
    pAttrL = [] # List of dictionaries (one per attribute) of attribute values
    ## and dictionaries of labels and probablities of that attribute given that label

        
    # Create the model by going through trainset to calculate the probability of
    # each label and the probabilities of each value of each attribute given each label
    def train(self, trainset):
        self.trainsize = len(trainset) # initialize the class variable
        
        # Start by counting up the number of instances with each label
        # I'll turn these counts into probabilities at the end
        lcounts = {}
        # and the number of instances with each attribute value and label
        valcounts = []
        # initialize as a list of dictionaries
        # will hold attribute values and dictionaries of labels and counts
        for a in range(0, len(self.attributes)):
            valcounts.append({})

        # loop through instances to do the counting
        for instance in trainset:
            # get the label and add it to lcounts
            l = instance[0]
            if l in lcounts:
                lcounts[l] += 1
            else:
                lcounts[l] = 1
            
            # loop through each attribute value and add to valcounts
            for a in range(1, len(self.attributes)):
                val = instance[a]
                # if there already is a dictionary in valcounts corresponding
                ## to the value and it contains a key corresponding to the label
                ### increment it, otherwise, add it.
                if val in valcounts[a]:
                    if l in valcounts[a][val]:
                        valcounts[a][val][l] += 1
                    else:
                        valcounts[a][val][l] = 1
                else:
                    valcounts[a][val] = {}
                    valcounts[a][val][l] = 1


        # now it's time to use the counts to calculate the probabilities

        # starting with the probabilitiy of each label
        # loop through each label and add its probability to pLabel
        for label in lcounts:
            self.pLabel[label] = lcounts[label]/self.trainsize

        # now to calculate the probability of each attribute value given each label
        # loop through valcounts to access the dictionary corresponding to each attribute
        for attrcounts in valcounts:
            # we'll need the number of values of each attribute for pseudocounts
            numvals = len(attrcounts)
            # create a new dictionary to add to pAttr|L, holding the probabilities
            pAttr = {}
            # then we can loop through attrcounts to access
            ## the dictionary corresponding to each value of that attribute
            for value in attrcounts:
                # create a new dictionary to add to pAttr
                pValL = {}
                # loop through all of the possible labels to calculate the probabilities
                for label in lcounts:
                    # we'll need the number of instances with this label
                    numl = lcounts[label]
                    # add the probability of value given label to pVal|L
                    if label in attrcounts[value]:
                        pValL[label] = (attrcounts[value][label] + 1)/(numl + numvals)
                    else:
                        # if there aren't any instances with that label and value, it'll have a small probability
                        pValL[label] = 1/(numl + numvals)
                # add pValL to pAttr under value
                pAttr[value] = pValL
            # add pAttr to pAttr|L
            self.pAttrL.append(pAttr)


    # test model on the testset and create a csv
    def test(self, testset, fname):
        # predict each item in the testset and count up the confusion matrix
        confusion = {}
        for instance in testset:
            reall = instance[0]
            # the actual prediction work
            predl = self.predict(instance)
            if reall not in confusion:
                confusion[reall] = {}
                confusion[reall][predl] = 1
            elif predl not in confusion[reall]:
                confusion[reall][predl] = 1
            else:
                confusion[reall][predl] += 1

        # output the confusion matrix as a csv file
        csv = ""
        # first row: predicted labels
        for label in self.pLabel:
            csv += label + ","
        csv += "\n"
        # loop through each real label to get each predicted label
        for label in self.pLabel:
            for label2 in self.pLabel:
                if label in confusion and label2 in confusion[label]:
                    csv += str(confusion[label][label2]) + ","
                else:
                    csv += str(0) + ","
            csv += label + "\n"
        file = open(fname, 'w')
        file.write(csv)

            
    # predict the label of an instance
    def predict(self, inst):
        pInstL = {}
        # go through each attribute in the instance and add the log of the
        ## probability of the value and attribute co-occuring
        for attr in range(1, len(inst)):
            val = inst[attr]
            for label in self.pLabel:
                if label not in pInstL:
                    pInstL[label] = 0
                if val in self.pAttrL[attr] and label in self.pAttrL[attr][val]:
                    pInstL[label] += math.log(self.pAttrL[attr][val][label])
                else:
                    pInstL[label] += math.log(1/(len(self.pAttrL[attr]) +
                                                 self.pLabel[label]*self.trainsize))
        # add log of the probability of each label to the probability of the instance
        ## to calculate the probability of each label and find the max to return
        best = 0
        bestL = ""
        for label in pInstL:
            pLInst = math.log(self.pLabel[label]) + pInstL[label]
            if 1/pLInst < best:
                best = 1/pLInst
                bestL = label
        return bestL



def main(filename, seednum):
    try:
        seednum = int(seednum)
        # turn the csv designated by filename into a trainset and testset

        model = NaiveBayes() # the model that will be trained
        
        # turn the file into a dataset
        dataset = []
        for line in open(filename):
            line = line.strip()
            if model.attributes == []:
                model.attributes = line.split(",")
            else:
                dataset.append(line.split(","))

        # shuffle dataset and split into test and train sets
        random.seed(seednum)
        random.shuffle(dataset)
        split = int(.2*len(dataset))
        testset = dataset[:split]
        trainset = dataset[split:]
        
        # train and test the model
        model.train(trainset)
        fname = "results_" + filename[:-4] + "_NaiveBayes_" + str(seednum) + ".csv"
        model.test(testset, fname)
    except OSError:
        print("Be sure to input a valid file name.")
    except ValueError:
        print("Be sure to input an integer seed.")

if __name__ == '__main__':
    filename = sys.argv[1]
    seednum = sys.argv[2]
    main(filename, seednum)
