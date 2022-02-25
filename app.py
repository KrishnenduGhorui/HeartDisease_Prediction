from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Loading trained ML model and Scaling object from pickle 

model = pickle.load(open('knnc_model.pickle', 'rb'))
sc_age = pickle.load(open('sc_age.pickle', 'rb'))
sc_trestbps = pickle.load(open('sc_trestbps.pickle', 'rb'))
sc_chol = pickle.load(open('sc_chol.pickle', 'rb'))
sc_thalach = pickle.load(open('sc_thalach.pickle', 'rb'))
sc_oldpeak = pickle.load(open('sc_oldpeak.pickle', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    # Taking input from front end page 
    if request.method == 'POST':
        age = int(request.form['age'])		
        cp = int(request.form['cp'])
        trestbps=float(request.form['trestbps'])
        chol = float(request.form['chol'])		
        fbs =int(request.form['fbs'])
        restecg =int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
       		
        sex=request.form['sex']
        if(sex=='Male'):            
            sex_male=1
        else:
            sex_male=0
        		
        exang=request.form['exang']
        if(exang=='Yes'):            
            exang_Yes=1
        else:
            exang_Yes=0
        # Standard Scaling some features
        ageScld=sc_age.transform(np.array(age).reshape(1,1))[0][0]
        trestbpsScld=sc_trestbps.transform(np.array(trestbps).reshape(1,1))[0][0]
        cholScld=sc_chol.transform(np.array(chol).reshape(1,1))[0][0]
        thalachScld=sc_thalach.transform(np.array(thalach).reshape(1,1))[0][0]
        oldpeakScld=sc_oldpeak.transform(np.array(oldpeak).reshape(1,1))[0][0]
		
        X=np.array([ageScld,sex_male,cp,trestbpsScld,cholScld,fbs,restecg,thalachScld,exang_Yes,oldpeakScld,slope,ca,thal]).reshape(1,-1)
        # Predicting target 
        prediction=model.predict(X)[0]
        		
        if (prediction==1):
          return render_template('index.html',prediction_text="Sorry, The patient has heart related disease")
        else:
          return render_template('index.html',prediction_text="Great, The patient has no heart related disease")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)