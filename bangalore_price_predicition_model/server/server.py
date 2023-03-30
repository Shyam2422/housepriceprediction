from flask import Flask, request, jsonify
import util
app = Flask(__name__)
@app.route('/get_location_names', methods=['GET'])



def get_location_names():
    util.load_saved_artifacts()
    # global __locations
    # return __locations
    response = jsonify({'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')


    return response

@app.route('/predict_home_price', methods=['GET','POST'])
def predict_home_price():

    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])



    response = jsonify({
        'estimated_price': util.get_estimated_price(location,total_sqft,bhk,bath)
    })

    response.headers.add('Access-Control-Allow-Origin','*')

    return response

if __name__ == "__main__":
    print("Starting Python flask server for bangalore home price prediction")
    util.load_saved_artifacts()
    app.run()


