from flask import Flask, render_template, request
import pickle
app = Flask(__name__)

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods = ["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        print(myDict)
        name = myDict['name']
        creditscore = int(myDict['creditscore'])
        tenure = int(myDict['tenure'])
        age = int(myDict['age'])
        balance = int(myDict['balance'])
        noofprod= int(myDict['noofprod'])
        esal = int(myDict['esal'])
        gender = int(myDict['gender'])
        isact = int(myDict['isactmem'])
        hascrcard = int(myDict['hascrcard'])
        geography = myDict['geography']
        if geography=='1,0,0':
            enter = [1,0,0,creditscore,gender,age,tenure,balance,noofprod,hascrcard,isact,esal]
        elif geography=='0,1,0':
            enter = [0,1,0,creditscore,gender,age,tenure,balance,noofprod,hascrcard,isact,esal]
        else:
            enter = [0, 0, 1, creditscore, gender, age, tenure, balance, noofprod, hascrcard, isact, esal]
        infProb = clf.predict([enter])[0]
        inf = abs(infProb)*100
        print(inf)
        return render_template('show.html',uname=name,inf=round(inf))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)