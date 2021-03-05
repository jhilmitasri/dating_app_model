from predictions import GetPredictions

class Run():
    def __init__(self):
        self.data_dict = {
            "person_1" : ["a", "a", "a", "a", "a", "a", "a", "b", "b", "b"],
            "person_2" : ["b", "b", "b", "b", "b", "a", "b", "b", "a", "b"],
            "person_3" : ["a", "a", "b", "b", "a", "a", "b", "a", "b", "a"],
            "person_4" : ["b", "b", "b", "b", "b", "b", "b", "b", "a", "b"],
        }
        self.cluster1 = []
        self.cluster2 = []
        self.predict = GetPredictions()
        results = self.predict.predict(self.data_dict)
        print("\nPrediction: {}\n".format(results))
        for person, cluster in results.items():
            if cluster == 0:
                self.cluster1.append(person)
            elif cluster == 1:
                self.cluster2.append(person)
        print("Got a similarity match for {}".format(self.cluster1))
        print("Got a similarity match for {}\n".format(self.cluster2))

if __name__ == "__main__":
    Run()
