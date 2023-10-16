import pickle
from flask import request, jsonify, Flask

app = Flask(__name__)

with open ('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)
    
@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    approval = y_pred >= 0.5
    
    result = {
        'credit_probability': y_pred,
        'approval': bool(approval)
    }
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
