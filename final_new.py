import pylab
import random
import math
import sklearn.linear_model

def variance(X):
    mean = sum(X)/len(X)
    tot = 0.0
    for x in X:
        tot += (x - mean)**2
    return tot/len(X)

def stdDev(X):
    return variance(X)**0.5

def getBMData(filename):
    """Read the contents of the given file. Assumes the file 
    in a comma-separated format, with 6 elements in each entry:
    0. Name (string), 1. Gender (string), 2. Age (int)
    3. Division (int), 4. Country (string), 5. Overall time (float)   
    Returns: dict containing a list for each of the 6 variables."""
    data = {}
    f = open(filename)
    line = f.readline() 
    data['cabin'], data['age'], data['gender'] = [], [], []
    data['survived'], data['name'] = [], []
    if 'class' in line.lower():
        line = f.readline()
    while line != '':
        split = line.strip().split(',', maxsplit=4)
        if len(split) < 5:
            line = f.readline()
            continue
        data['cabin'].append(int(split[0]))
        data['age'].append(float(split[1]))
        data['gender'].append(split[2].strip().upper())
        data['survived'].append(int(split[3]))
        data['name'].append(split[4].strip())
        line = f.readline()
    f.close()
    # maleTime, femaleTime = [], []
    # for i in range(len(data['time'])):
    #     if data['gender'][i]=='M':
    #         maleTime.append(data['time'][i])
    #     else:
    #         femaleTime.append(data['time'][i])
    # print(len(maleTime),' Males and', len(femaleTime),'Females')    
    # return data, maleTime, femaleTime
    return data

class Passenger(object): 
    def __init__ (self, cabin, age, gender, survived): 
        self.featureVec = (cabin, age, gender) 
        self.label = survived 

    def featureDist(self, other): 
        dist = 0.0 
        for i in range(len(self.featureVec)): 
            dist += abs(self.featureVec[i] - other.featureVec[i])**2 
        return dist**0.5 
    
    def cosine_similarity(self,other):
    #compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(self.featureVec)):
            x = self.featureVec[i]; y = other.featureVec[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)

    def getCabin(self): 
        return self.featureVec[0] 
    
    def getAge(self): 
        return self.featureVec[1] 
    
    def getGender(self): 
        return self.featureVec[2]

    def getLabel(self): 
        return self.label 

    def getFeatures(self): 
        return self.featureVec 

    # def __str__ (self): 
    #     return str(self.getAge()) + ', ' + str(self.getTime()) + ', ' + self.label

def buildTitanicExamples(fileName): 
    # data, maleTime, FemaleTime = getBMData(fileName)
    data = getBMData(fileName)
    examples = [] 
    for i in range(len(data['age'])): 
        a = Passenger(data['cabin'][i], data['age'][i], data['gender'][i], data['survived'][i]) 
        examples.append(a)
    return examples 
 

def makeHist(data_all, data_s, data_n, bins, title, xLabel, yLabel, gender):
    all = [p.getAge() for p in data_all]
    survived_ages = [p.getAge() for p in data_s]
    not_survived_ages = [p.getAge() for p in data_n]
    mean_survived = sum(survived_ages)/len(survived_ages)
    std_survived = stdDev(survived_ages)
    mean_all = sum(all)/len(all)
    std_all = stdDev(all)
    pylab.figure(figsize=(8, 5))
    pylab.hist([survived_ages, not_survived_ages], 
        bins=bins, stacked=True,
        label=[
            f'Survived {gender} Passengers\nMean={mean_survived:.2f}, SD={std_survived:.2f}', 
            f'All {gender} Passengers\nMean={mean_all:.2f}, SD={std_all:.2f}'
        ],
        edgecolor='black', color=['darkorange', 'steelblue'])
    pylab.xlabel(xLabel)
    pylab.ylabel(yLabel)
    pylab.title(title)
    pylab.legend()
    pylab.tight_layout()
    pylab.show()

def makeClassHist(data_s, data_n, title, xLabel, yLabel):
    survived_classes = [p.getCabin() for p in data_s if isinstance(p.getCabin(), (int, float))]
    not_survived_classes = [p.getCabin() for p in data_n if isinstance(p.getCabin(), (int, float))]
    pylab.figure(figsize=(8, 5))
    pylab.hist([survived_classes, not_survived_classes], bins=[0.5, 1.5, 2.5, 3.5], 
        stacked=True, edgecolor='black', color=['darkorange', 'steelblue'])
    pylab.xticks([1, 2, 3])
    pylab.xlabel(xLabel)
    pylab.ylabel(yLabel)
    pylab.title(title)
    pylab.tight_layout()
    pylab.show()

def gender_split(data):    
    female_data = []
    male_data = []
    for instance in data:
        if instance.getGender() == 'F':
            female_data.append(instance)
        elif instance.getGender() == 'M':
            male_data.append(instance)
    return male_data, female_data

def survived_split(data):
    survived_data = []
    not_survived_data = []
    for instance in data:
        if instance.getLabel() == 1:
            survived_data.append(instance)
        elif instance.getLabel() == 0:
            not_survived_data.append(instance)
    return survived_data, not_survived_data

def data_convert(data):
    X = [] 
    y = []
    for p in data:
        C1 = 1 if p.getCabin() == 1 else 0
        C2 = 1 if p.getCabin() == 2 else 0
        C3 = 1 if p.getCabin() == 3 else 0
        # M=1, F=0
        gender_val = 1 if p.getGender() == 'M' else 0
        age = p.getAge()
        feature = [C1, C2, C3, age, gender_val]
        X.append(feature)
        y.append(p.getLabel())
    return X, y

def divide80_20(examples): 
    sampleIndices = random.sample(range(len(examples)), len(examples)//5) 
    trainingSet, testSet = [], [] 
    for i in range(len(examples)): 
        if i in sampleIndices: 
            testSet.append(examples[i]) 
        else: trainingSet.append(examples[i]) 
    return trainingSet, testSet 

def mean_confidence_interval(data):
    mean = sum(data)/len(data)
    std = stdDev(data)
    # ci = 1.96 * (std / len(data)**0.5)
    ci = 1.96 * std
    return mean, ci

def accuracy(truePos, falsePos, trueNeg, falseNeg): 
    numerator = truePos + trueNeg 
    denominator = truePos + trueNeg + falsePos + falseNeg 
    return numerator/denominator 
def sensitivity(truePos, falseNeg): 
    try: 
        return truePos/(truePos + falseNeg) 
    except ZeroDivisionError: 
        return float('nan') 
def specificity(trueNeg, falsePos): 
    try: 
        return trueNeg/(trueNeg + falsePos) 
    except ZeroDivisionError: 
        return float('nan') 
def posPredVal(truePos, falsePos): 
    try:
        return truePos/(truePos + falsePos) 
    except ZeroDivisionError: 
        return float('nan') 
def negPredVal(trueNeg, falseNeg): 
    try: 
        return trueNeg/(trueNeg + falseNeg) 
    except ZeroDivisionError: 
        return float('nan') 

def applyModel(model, test_F, test_L, label, prob ):
    #Create vector containing feature vectors for all test examples
    testFeatureVecs = test_F
    probs = model.predict_proba(testFeatureVecs)
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(probs)):
        if probs[i][1] > prob:
            if test_L[i] == label:
                truePos += 1
            else:
                falsePos += 1
        else:
            if test_L[i] != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg

def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint): 
    accur = accuracy(truePos, falsePos, trueNeg, falseNeg) 
    sens = sensitivity(truePos, falseNeg) 
    spec = specificity(trueNeg, falsePos) 
    ppv = posPredVal(truePos, falsePos) 
    if toPrint: 
        print(' Accuracy =', round(accur, 3)) 
        print(' Sensitivity =', round(sens, 3)) 
        print(' Specificity =', round(spec, 3)) 
        print(' Pos. Pred. Val. =', round(ppv, 3)) 
    return (accur, sens, spec, ppv)

def compute_auroc(y_true, y_scores, pos_label=1):
    paired = list(zip(y_true, y_scores))
    paired.sort(key=lambda x: -x[1])
    TPRs, FPRs = [], []
    P = sum(1 for y in y_true if y == pos_label)
    N = len(y_true) - P
    tp = fp = 0
    for i, (label, score) in enumerate(paired):
        if label == pos_label:
            tp += 1
        else:
            fp += 1
        if i == len(paired) - 1 or score != paired[i + 1][1]:
            TPRs.append(tp / P if P else 0)
            FPRs.append(fp / N if N else 0)
    auroc = 0.0
    for i in range(1, len(TPRs)):
        auroc += (FPRs[i] - FPRs[i - 1]) * (TPRs[i] + TPRs[i - 1]) / 2
    return auroc


def run_1000_trials(examples):
    weights_list = [[], [], [], [], []]
    intercepts, accuracies, sensitivities, specificities, ppvs, aurocs = [], [], [], [], [], []

    for i in range(1000):
        train, test = divide80_20(examples)
        X_train, y_train = data_convert(train)
        X_test, y_test = data_convert(test)
        model = sklearn.linear_model.LogisticRegression().fit(X_train, y_train)
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, X_test, y_test, 1, 0.5)
        y_probs = list(model.predict_proba(X_test)[:, 1])
        auroc = compute_auroc(y_test, y_probs)
        aurocs.append(auroc)
        accur, sens, spec, ppv = getStats(truePos, falsePos, trueNeg, falseNeg, toPrint=False)
        accuracies.append(accur)
        sensitivities.append(sens)
        specificities.append(spec)
        ppvs.append(ppv)
        coef = model.coef_[0]
        for i in range(len(coef)):
            weights_list[i].append(coef[i])
        intercepts.append(model.intercept_[0])

    results = {}
    feature_names = ['C1', 'C2', 'C3', 'age', 'gender']
    for i in range(5):
        mean, ci = mean_confidence_interval(weights_list[i])
        results[f'Mean weight of {feature_names[i]}'] = (mean, ci)
    results['Mean intercept of fitted model'] = mean_confidence_interval(intercepts)
    results['Mean accuracy'] = mean_confidence_interval(accuracies)
    results['Mean sensitivity'] = mean_confidence_interval(sensitivities)
    results['Mean specificity'] = mean_confidence_interval(specificities)
    results['Mean pos.pred.val.'] = mean_confidence_interval(ppvs)
    results['Mean AUROC'] = mean_confidence_interval(aurocs)
    return results




# --- main ---
examples = buildTitanicExamples('TitanicPassengers.txt')
male_data, female_data = gender_split(examples)
survived_mdata, not_survived_mdata = survived_split(male_data)
survived_fdata, not_survived_fdata = survived_split(female_data)

# Survived Male/Female Passengers VS Ages
makeHist(male_data, survived_mdata, not_survived_mdata, 20, 'Survived Male Passengers VS Ages'
         , 'Male Ages', 'Number of Male Passengers', 'Male')
makeHist(female_data, survived_fdata, not_survived_fdata, 20, 'Survived Female Passengers VS Ages'
         , 'Female Ages', 'Number of Female Passengers', 'Female')

# Male/Female Cabin Classes VS Survived
makeClassHist(survived_mdata, not_survived_mdata, 'Male Cabin Classes VS Survived'
              , 'Male Cabin Classes', 'Number of Male Passengers')
makeClassHist(survived_fdata, not_survived_fdata, 'Female Cabin Classes VS Survived'
              , 'Female Cabin Classes', 'Number of Female Passengers')

# 3
run_results = run_1000_trials(examples)
print("Logistic Regression:\nAverages for all examples 1000 trials with threshold k=0.5")
for key, (mean, ci) in run_results.items():
    print(f" {key} = {mean:.3f}, 95% confidence interval = {ci:.3f}")


