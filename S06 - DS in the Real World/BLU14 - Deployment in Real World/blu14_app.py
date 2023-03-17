import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


########################################
# Begin database stuff

DB = SqliteDatabase('prediction.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    data = TextField()
    probability = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)


with open('pipeline.pickle', 'rb') as fh:
    pipeline = joblib.load(fh)


with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Input validation functions


def check_request(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "observation_id" not in request:
        error = 'observation_id is missing'
        return False, error
    
    if "data" not in request:
        error = 'data is missing'
        return False, error
    
    return True, ''

def check_valid_column(data):
    """
        Validates that our data only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = {'age', 'workclass', 'education', 'marital-status', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week'}
    
    keys = set(data.keys())
    
    if len(valid_columns - keys) > 0:
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error

    return True, ''


def check_categorical_values(data):
    """
        Validates that all categorical fields are in the data and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_category_map = {
        "sex": ["Male", "Female"],
        "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
    }

    for key, valid_categories in valid_category_map.items():
        if key in data:
            value = data[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error

    return True, ""

def check_age(data):
    """
        Validates that data contains valid age values
        
        Returns:
        - assertion value: True if hour is valid, False otherwise
        - error message: empty if hour is valid, False otherwise
    """
    
    age = data.get("age")
        
    if not isinstance(age, int):
        error = "Field `age` is not an integer"
        return False, error
    
    if age < 10 or age > 100:
        error = "Invalid value provided for age: {}".format(age)
        return False, error

    return True, ""



def check_capital_gain_loss(data):
    """
        Validates that data contains valid capital-gain and capital-loss valid values 
        
        Returns:
        - assertion value: True if hour is valid, False otherwise
        - error message: empty if hour is valid, False otherwise
    """
    
    gain = data.get("capital-gain")
    loss = data.get("capital-loss")
        

    if gain < 0 or gain > 99999:
        error = "Invalid value provided for capital-gain: {}".format(gain)
        return False, error

    if loss < 0 or loss > 4356:
        error = "Invalid value provided for capital-loss: {}".format(loss)
        return False, error

    return True, ""


def check_hour_week(data):
    """
        Validates that data contains valid hours per week
        
        Returns:
        - assertion value: True if hour is valid, False otherwise
        - error message: empty if hour is valid, False otherwise
    """
    
    hour = data.get("hours-per-week")
        
    
    if hour < 0 or hour > 168:
        error = "Invalid value provided for hours-per-week: {}".format(hour)
        return False, error

    return True, ""

# End input validation functions
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
  
    request_ok, error = check_request(obs_dict)
    if not request_ok:
        response = {'observation_id': None, 'error': error}
        return jsonify(response)

    _id = obs_dict['observation_id']
    data = obs_dict['data']

    columns_ok, error = check_valid_column(data)
    if not columns_ok:
        response = {'observation_id': _id, 'error': error}
        return jsonify(response)

    categories_ok, error = check_categorical_values(data)
    if not categories_ok:
        response = {'observation_id': _id, 'error': error}
        return jsonify(response)

    age_ok, error = check_age(data)
    if not age_ok:
        response = {'observation_id': _id, 'error': error}
        return jsonify(response)
    
    capital_ok, error = check_capital_gain_loss(data)
    if not capital_ok:
        response = {'observation_id': _id,'error': error}
        return jsonify(response)

    hour_ok, error = check_hour_week(data)
    if not hour_ok:
        response = {'observation_id': _id,'error': error}
        return jsonify(response)



    obs = pd.DataFrame([data], columns=columns).astype(dtypes)
    probability = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response = {'observation_id': _id, 'prediction': bool(prediction), 'probability': probability}
    p = Prediction(
        observation_id=_id,
        probability=probability,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)

    
@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})


    
if __name__ == "__main__":
    app.run(debug=True)
