from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS
import os
# import request
# import json
from modul import Module
app = Flask(__name__)

# init object flask
app = Flask(__name__)

# init object flask restfull
api = Api(app)

# init cors
CORS(app)


@app.route('/')
def gas():
    return 'Connected'


@app.route('/recommendation', methods=["POST"])
def main():
    res = {}

    lat = request.form('latitude')
    longit = request.form('longitude')

    modul = Module('materials.xlsx')
    modul.updateDistance(lat, longit)

    arr = request.form('criteria')
    data = arr.split(",")
    res["result"] = modul.getBobotKriteria(data)
    return res


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
