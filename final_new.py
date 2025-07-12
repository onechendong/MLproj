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

def makeAccurHist(data, title, xLabel, yLabel, label):
    mean = sum(data) / len(data)
    std = stdDev(data)
    pylab.figure(figsize=(8, 5))
    pylab.hist(data, bins=15, edgecolor='black', color='steelblue', 
               label=[
            f'{label}\nMean = {mean:.2f}, SD={std:.2f}'
        ])
    pylab.xlabel(xLabel)
    pylab.ylabel(yLabel)
    pylab.title(title)
    pylab.legend()
    pylab.tight_layout()
    pylab.show()

def plot_mean_accuracy_vs_k(ks, mean_accuracies, title):
    best_mean_acc = max(mean_accuracies)
    best_k = ks[mean_accuracies.index(best_mean_acc)]
    pylab.figure(figsize=(8, 5))
    pylab.plot(ks, mean_accuracies, color='steelblue', label='Mean Accuracies of thresholds ks(0.5, 0.65)')
    pylab.scatter([best_k], [best_mean_acc], color='red', label='Maximum Mean Accuracy')
    pylab.annotate(f'({best_k:.3f}, {best_mean_acc:.3f})',xy=(best_k, best_mean_acc),
                   xytext=(best_k + 0.003, best_mean_acc + 0.000),fontsize=10)    
    pylab.title(title)
    pylab.xlabel('Threshold Value ks between 0.5 to 0.65')
    pylab.ylabel('Accuracy')
    pylab.legend()
    pylab.tight_layout()
    pylab.show()
    return best_k

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

def plot_mean_roc_curve(mean_weights, mean_intercept, examples, title):
    X_test, y_test = data_convert(examples)
    y_scores = []
    for features in X_test:
        linear_comb = sum(w * x for w, x in zip(mean_weights, features)) + mean_intercept
        prob = 1 / (1 + math.exp(-linear_comb))
        y_scores.append(prob)
    thresholds = sorted(set(y_scores), reverse=True)
    TPRs, FPRs = [], []
    P = sum(1 for y in y_test if y == 1)
    N = len(y_test) - P
    for thresh in thresholds:
        tp = fp = 0
        for i in range(len(y_test)):
            if y_scores[i] >= thresh:
                if y_test[i] == 1:
                    tp += 1
                else:
                    fp += 1
        TPR = tp / P if P else 0
        FPR = fp / N if N else 0
        TPRs.append(TPR)
        FPRs.append(FPR)
    auroc = 0.0
    for i in range(1, len(TPRs)):
        auroc += (FPRs[i] - FPRs[i - 1]) * (TPRs[i] + TPRs[i - 1]) / 2

    pylab.figure(figsize=(8, 5))
    pylab.plot(FPRs, TPRs, color='steelblue')
    pylab.plot([0, 1], [0, 1], linestyle='--', color='orange')
    pylab.xlabel('1 - Specificity - False Positive Rate')
    pylab.ylabel('Sensitivity - True Positive Rate')
    pylab.title(f'{title} (AUROC = {auroc:.3f})')
    pylab.tight_layout()
    pylab.show()

def zScaleExamples(examples):
    ages = [p.getAge() for p in examples]
    mean_age = sum(ages) / len(ages)
    std_age = stdDev(ages)
    scaled_data = []
    for p in examples:
        cabin = p.getCabin()
        gender = p.getGender()
        label = p.getLabel()
        age = p.getAge()
        z_scaled_age = (age - mean_age) / std_age
        scaled_p = Passenger(cabin, z_scaled_age, gender, label)
        scaled_data.append(scaled_p)
    return scaled_data

def iScaleExamples(examples):
    ages = [p.getAge() for p in examples]
    min_age = min(ages)
    max_age = max(ages)
    scaled_data = []
    for p in examples:
        cabin = p.getCabin()
        gender = p.getGender()
        label = p.getLabel()
        age = p.getAge()
        i_scaled_age = (age - min_age) / (max_age - min_age)
        scaled_p = Passenger(cabin, i_scaled_age, gender, label)
        scaled_data.append(scaled_p)
    return scaled_data

def run_1000_trials(examples):
    weights_list = [[], [], [], [], []]
    intercepts, accuracies, sensitivities, specificities, ppvs, aurocs = [], [], [], [], [], []
    ks = [round(0.5 + t * (0.65 - 0.5) / 999, 5) for t in range(1000)]
    accuracies_max, best_k_value = [], []
    accuracy_per_k = [[] for _ in range(1000)]
    mean_accuracy_per_k = []
    mean_weights = []

    for i in range(1000):
        train, test = divide80_20(examples)
        X_train, y_train = data_convert(train)
        X_test, y_test = data_convert(test)
        model = sklearn.linear_model.LogisticRegression().fit(X_train, y_train)
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, X_test, y_test, 1, 0.5)

        accuracies_diff_k = []
        for idx, k in enumerate(ks):
            truePos_k, falsePos_k, trueNeg_k, falseNeg_k = applyModel(model, X_test, y_test, 1, k)
            accur_diff = accuracy(truePos_k, falsePos_k, trueNeg_k, falseNeg_k)
            accuracies_diff_k.append(accur_diff)
            accuracy_per_k[idx].append(accur_diff) # idx=0 --> k=0.5, idx=999 --> k=0.65
        max_accur = max(accuracies_diff_k)
        best_k = ks[accuracies_diff_k.index(max_accur)]
        accuracies_max.append(max_accur)
        best_k_value.append(best_k)

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

    for acc_list in accuracy_per_k:
        mean_accuracy_per_k.append(sum(acc_list)/len(acc_list))

    results = {}
    feature_names = ['C1', 'C2', 'C3', 'age', 'gender']
    for i in range(5):
        mean, ci = mean_confidence_interval(weights_list[i])
        mean_weights.append(mean)
        results[f'Mean weight of {feature_names[i]}'] = (mean, ci)
    mean_intercept, ci_intercept = mean_confidence_interval(intercepts)
    results['Mean intercept of fitted model'] = mean_intercept, ci_intercept
    results['Mean accuracy'] = mean_confidence_interval(accuracies)
    results['Mean sensitivity'] = mean_confidence_interval(sensitivities)
    results['Mean specificity'] = mean_confidence_interval(specificities)
    results['Mean pos.pred.val.'] = mean_confidence_interval(ppvs)
    results['Mean AUROC'] = mean_confidence_interval(aurocs)
    return results, accuracies_max, best_k_value, mean_accuracy_per_k, ks, mean_weights, mean_intercept




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
run_results, accuracies_max, best_k_value, mean_accuracy_per_k, ks, mean_weights, mean_intercept = run_1000_trials(examples)
print("\nLogistic Regression:\nAverages for all examples 1000 trials with threshold k=0.5")
for key, (mean, ci) in run_results.items():
    print(f" {key} = {mean:.3f}, 95% confidence interval = {ci:.3f}")

makeAccurHist(accuracies_max, 'Maximum Accuracies', 'Maximum Accuracies', 
              'Numbers of Maximum Accuracies', 'Maximum Accuracies for 1000 splits')
makeAccurHist(best_k_value, 'Threshold Values k for Maximum Accuracies', 
              'Thresholds Values ks Between 0.5 and 0.65', 'Numbers of ks', 'k Values for Maximum Accuracies')
_ = plot_mean_accuracy_vs_k(ks, mean_accuracy_per_k, 'Mean Accuracy for Different Threshold k Values')
plot_mean_roc_curve(mean_weights, mean_intercept, examples, 'ROC with Mean Weights and Intercepts')

# 4 zScaling
z_examples = zScaleExamples(examples)
z_run_results, z_accuracies_max, z_best_k_value, z_mean_accuracy_per_k, z_ks, z_mean_weights, z_mean_intercept = run_1000_trials(z_examples)
print("\nLogistic Regression with zScaling:\nAverages for all examples (zScaling) 1000 trials with threshold k=0.5")
for key, (mean, ci) in z_run_results.items():
    print(f" {key} = {mean:.3f}, 95% confidence interval = {ci:.3f}")
makeAccurHist(z_accuracies_max, '(zScaling) Maximum Accuracies', 'Maximum Accuracies', 
              'Numbers of Maximum Accuracies', 'Maximum Accuracies for 1000 splits')
makeAccurHist(z_best_k_value, '(zScaling) Threshold Values k for Maximum Accuracies', 
              'Thresholds Values ks Between 0.5 and 0.65', 'Numbers of ks', 'k Values for Maximum Accuracies')
_ = plot_mean_accuracy_vs_k(z_ks, z_mean_accuracy_per_k, '(zScaling) Mean Accuracy for Different Threshold k Values')
plot_mean_roc_curve(z_mean_weights, z_mean_intercept, z_examples, '(zScaling) ROC with Mean Weights and Intercepts')

# 4 iScaling
i_examples = iScaleExamples(examples)
i_run_results, i_accuracies_max, i_best_k_value, i_mean_accuracy_per_k, i_ks, i_mean_weights, i_mean_intercept = run_1000_trials(i_examples)
print("\nLogistic Regression with iScaling:\nAverages for all examples (iScaling) 1000 trials with threshold k=0.5")
for key, (mean, ci) in i_run_results.items():
    print(f" {key} = {mean:.3f}, 95% confidence interval = {ci:.3f}")
makeAccurHist(i_accuracies_max, '(iScaling) Maximum Accuracies', 'Maximum Accuracies', 
              'Numbers of Maximum Accuracies', 'Maximum Accuracies for 1000 splits')
makeAccurHist(i_best_k_value, '(iScaling) Threshold Values k for Maximum Accuracies', 
              'Thresholds Values ks Between 0.5 and 0.65', 'Numbers of ks', 'k Values for Maximum Accuracies')
i_best_k = plot_mean_accuracy_vs_k(i_ks, i_mean_accuracy_per_k, '(iScaling) Mean Accuracy for Different Threshold k Values')
plot_mean_roc_curve(i_mean_weights, i_mean_intercept, i_examples, '(iScaling) ROC with Mean Weights and Intercepts')
print(f'(iScaling) statistics for mean maximum threshold k= {i_best_k:.3f}')