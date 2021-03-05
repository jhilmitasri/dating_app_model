# dating_app_model

A Machine Learning implementation of sorting dating profiles based on a short questionnaire 

## Installation

Use the package manager [pip3](https://pip.pypa.io/en/stable/) to install libraries for the dating_app_model.


```bash
pip3 install -r requirements.txt
```

## Implimentation

```bash
python3 run.py
```
## Run your own data

To run the model on your data, you can go inside run.py and change the following dictionary.
This dictionary indicates the 10 answers to questions in the given questionnaire.
 
```python3
self.data_dict = {
 "person_1" : ["a", "a", "a", "a", "a", "a", "a", "b", "b", "b"],
 "person_2" : ["b", "b", "b", "b", "b", "a", "b", "b", "a", "b"],
 "person_3" : ["a", "a", "b", "b", "a", "a", "b", "a", "b", "a"],
 "person_4" : ["b", "b", "b", "b", "b", "b", "b", "b", "a", "b"],
        }
```