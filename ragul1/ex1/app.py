from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np 

app = Flask(__name__)


model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')


@app.router("/predict", method=['POST'])
def predict():
    float_features = [ float(x) for x in request.form.value()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template('index.html',prediction_text='The Flower specifes is: {}'.format(prediction))

                           
if __name__ == '__main__':
    app.run(debug=true) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
                                          