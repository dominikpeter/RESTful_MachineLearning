from flask import Flask
from flask import render_template, request, url_for, redirect
from flask import jsonify
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
import numpy as np
import bs4 as bs
import re
import urllib.request

app = Flask(__name__)

@app.route("/")
def hello():
    return redirect("/landing")

@app.route("/testing")
def testing():
    return render_template('page_layout.html')

@app.route("/landing")
def vogel():
    return render_template('landing_page.html')

@app.route("/sinus")
def sinus():
    return render_template('sinus.html')

@app.route("/barchart")
def bar():
    return render_template('barchart.html')

@app.route('/svm')
def svm():
    digits = load_digits()
    clf = SVC(gamma=0.001, C=100.)
    clf.fit(digits.data[:-1], digits.target[:-1])
    prediction = clf.predict(digits.data[-1:])

    return jsonify({'prediction': repr(prediction[0])})


@app.route("/housing")
def form():
    return render_template('form_submit.html')


@app.route('/predict_housing/', methods=['POST'])
def predict_housing():
    age=request.form['age']
    rooms=request.form['rooms']
    bedrooms=request.form['bedrooms']

    age=float(age)
    rooms=float(rooms)
    bedrooms=float(bedrooms)

    housing = fetch_california_housing()
    data = housing['data']
    data = data[:,1:4]
    target = housing['target']

    reg = linear_model.Ridge(alpha = .8)

    reg.fit(data, target)

    newdata = np.asarray([age, rooms, bedrooms])

    prediction=reg.predict(newdata.reshape(1,-1))
    prediction=round(float(prediction),2)*1000000

    return render_template('form_prediction.html', prediction=prediction)

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')

@app.route('/_add_sentiment')
def add_sentiment():
    url = request.args.get('a', 0, type=str)
    url = str('https://www.' + re.sub('https://|www.', "", url))
    source = urllib.request.urlopen(url)
    soup = bs.BeautifulSoup(source, 'html.parser', parse_only=bs.SoupStrainer('div'))
    txt = soup.text
    txt = re.findall("[A-z]+", txt)
    txt = " ".join(txt)
    return jsonify(result=txt)


if __name__ == "__main__":
    app.debug = True
    app.run()
