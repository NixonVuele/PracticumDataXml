from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
# Lectura de los modelos del algoritmo de native Bayes
vectorizerNB = pickle.load(open('Modelos/vectorizerBN.sav', 'rb'))
classifierNB = pickle.load(open('Modelos/classifierNB.sav', 'rb'))

# Lectura de los modelos del algoritmo de KNN Edificio 5 Sala i
vectorizerKNN = pickle.load(open('Modelos/vectorizerKNN.sav', 'rb'))
classifierKNN = pickle.load(open('Modelos/classifierKNN.sav', 'rb'))

# Lectura de los modelos del algoritmo de SMV

vectorizerSVM = pickle.load(open('Modelos/vectorizerSVM.sav', 'rb'))
classifierSVM = pickle.load(open('Modelos/classifierSVM.sav', 'rb'))

@app.route('/', methods=["GET","POST"])
def main():
    # Algoritmo de native Bayes
    if request.method == "POST" and request.form.get("submit1") == "Send NB":
        inp = request.form["entrada1"]
        text_vectorNB = vectorizerNB.transform([inp])
        resultNB = classifierNB.predict(text_vectorNB)
        if resultNB[0] == 1:
            return render_template("home.html",message ="Negative π­π­")
        elif resultNB[0] == 3:
            return render_template("home.html",message = "Positive ππ")
        elif resultNB[0] == 2:
            return render_template("home.html",message = "Neutro ππ")
        elif resultNB[0] == 0:
            return render_template("home.html", message="None πΆπΆ")

    if request.method == "POST" and request.form.get("submit2") == "Send KNN":
        # Algoritmo de KKNVecinos
        inp2 = request.form["entrada2"]
        text_vectorKNN = vectorizerKNN.transform([inp2])
        resultKNN = classifierKNN.predict(text_vectorKNN)
        if resultKNN[0] == 1:
            return render_template("home.html", message2="Negative π­π­")
        elif resultKNN[0] == 3:
             return render_template("home.html", message2="Positive ππ")
        elif resultKNN[0] == 2:
            return render_template("home.html", message2="Neutro ππ")
        elif resultKNN[0] == 0:
            return render_template("home.html", message3="None πΆπΆ")

    if request.method == "POST" and request.form.get("submit3") == "Send SVM":
        inp3 = request.form.get("entrada3")
        text_vectorSVM = vectorizerSVM.transform([inp3])
        resultSVM = classifierSVM.predict(text_vectorSVM)
        if resultSVM[0] == 'N':
            return render_template("home.html", message3="Negative π­π­")
        elif resultSVM[0] == 'P':
            return render_template("home.html", message3="Positive ππ")
        elif resultSVM[0] == 'NEU':
            return render_template("home.html", message3="Neutro ππ")
        elif resultSVM[0] == 'NONE':
            return render_template("home.html", message3="None πΆπΆ")
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)