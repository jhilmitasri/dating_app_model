# testlists = []
# for i in range(0,11):
#     testlist = []
#     for i in range(0,i):
#         testlist.append(0)
#     for i in range(0,10-len(testlist)):
#         testlist.append(1)
    
#     print(testlist, len(testlist), 10-len(testlist))
#     testlists.append(testlist)

# print(testlists, (len(testlists)))



from sklearn import preprocessing
import numpy as np

labelmaker = preprocessing.LabelBinarizer()
dp_a = labelmaker.fit_transform(["a", "a", "a", "a", "a", "a", "b", "b", "b", "b"])
dp_b = labelmaker.fit_transform(["b", "b", "b", "b", "b", "b", "b", "b", "b", "b"])
print(type(dp_a), np.reshape(dp_b, (1,10)))