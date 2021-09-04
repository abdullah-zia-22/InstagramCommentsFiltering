from flask import Flask, render_template, request,jsonify
from flask_cors import CORS, cross_origin
import pickle
import json
import ast

app = Flask(__name__, template_folder='templates')
cors = CORS(app)

app.config['EXPLAIN_TEMPLATE_LOADING'] = True
app.config['CORS_HEADERS'] = "Content-Type"


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    category = ['Neither','Offensive','Hate Speech']
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))

    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    data = request.form
    sentence = data['sentence']

    sentence=[sentence]
    text_features = vectorizer.transform(sentence)

    predictions = loaded_model.predict(text_features)

    predictions=("Predicted as: '{}'".format(category[predictions[0]]))
    return render_template('index.html', predicted_text=predictions)

#remove GET method
@app.route('/comments_prediction', methods=['POST'])
@cross_origin()
def comments_prediction():
    all_comments = request.json
    #print(all_comments)
    category = ['Neither', 'Offensive', 'Hate Speech']
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))

    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    #converting json into python dictionary
    #json_data="[{'id': 1, 'text': 'MainMan do a reaction video of Heihachi vs Geese DEATH BATTLE.','prediction': 'None'}, {'id': 2, 'text': 'fuck you','prediction': 'None'},{'id': 3, 'text': 'nigga','prediction': 'None'}]"
    #json_data = ast.literal_eval(all_comments)
    python_dic = json.dumps(all_comments)
    #python_dic=ast.literal_eval(str(python_dic))
    python_dic=json.loads(python_dic)
    #print(python_dic[0]["text"])

    for x in range(len(python_dic)):
        text_features = vectorizer.transform([python_dic[x]["text"]])

        predictions = loaded_model.predict(text_features)

        #predictions = ("Predicted as: '{}'".format(category[predictions[0]]))
        #print(category[predictions[0]])
        if(predictions[0]==1 or predictions[0]==2):
            python_dic[x]['prediction']=category[predictions[0]]

    #python dictionary to json array to send json to foreground.js
    return jsonify(python_dic)






if __name__ == "__main__":
    app.run(debug=True)
