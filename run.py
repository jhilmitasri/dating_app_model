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
        self.svc_predict = GetPredictions()
        results = self.svc_predict.predict(self.data_dict)
        people_list = results.keys()
        covered_person = []
        matching_probabilities = {}
        for person in results.keys():
            for listed_person in people_list:
                if(person != listed_person and listed_person not in covered_person):
                    matching_probabilities["{} - {}".format(person, listed_person)] = 100 \
                                    - (abs(results[person][0] - results[listed_person][0])*100 
                                    + abs(results[person][1] - results[listed_person][1])*100)
                    covered_person.append(person)
        for match, probability in matching_probabilities.items():
            print("\n\t{} ----> {}%".format(match, round(probability, 2)))
        

if __name__ == "__main__":
    Run()
