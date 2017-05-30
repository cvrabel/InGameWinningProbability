from flask import Flask, render_template, redirect, url_for, request
from flask import make_response
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from enum import Enum
import pandas as pd
import math
import classifier3
import os
import simplejson as json
import pickle


predict_script = Flask(__name__)
predict_script.debug=True

# class Action(Enum):
# 	push = 0
# 	homeMake = homeTurnover = homeFoulNonS = homeFreeMakeLast = awayRebound = awayJumpBall = awayTimeout = -1.088
# 	awayMake = awayTurnover = awayFoulNonS = awayFreeMakeLast = homeRebound = homeJumpBall = homeTimeout = 1.088
# 	homeMiss = homeFreeMiss = -0.8323
# 	awayMiss = awayFreeMiss = 0.8323
# 	homeFt1 = 0.7645
# 	homeFt2 = 1.529
# 	homeFt3 = 2.294
# 	awayFt1 = -0.7645
# 	awayFt2 = -1.529
# 	awayFt3 = -2.294

class Action(Enum):
	push = 0
	homeMake = homeTurnover = homeFoulNonS = homeFreeMakeLast = awayRebound = awayJumpBall = awayTimeout = -0.95
	awayMake = awayTurnover = awayFoulNonS = awayFreeMakeLast = homeRebound = homeJumpBall = homeTimeout = 0.95
	homeMiss = homeFreeMiss = -0.7266
	awayMiss = awayFreeMiss = 0.7266
	homeFt1 = 0.7645
	homeFt2 = 1.529
	homeFt3 = 2.294
	awayFt1 = -0.7645
	awayFt2 = -1.529
	awayFt3 = -2.294

def eventParse(event):
	if "Home Make" in str(event):
		return Action.homeMake.value
	elif "Home Miss" in str(event):
		return Action.homeMiss.value
	return 0



@predict_script.route("/", methods=['GET', 'POST'])
@predict_script.route('/predict', methods=['GET', 'POST'])
def predict():
	message = None
	with open('classifier_score_clutch_adj_2_KNN_500.pkl', 'rb') as f:
		classifier = pickle.load(f)

	if request.method == 'POST':
		clock = float(request.form['myclock'])
		home = float(request.form['myhome'])
		away = float(request.form['myaway'])
		evt = request.form['myevent']
		
		event = eval("Action."+evt+".value")
		event = event * classifier3.clutchAdj(clock)
		score = home - away + event
		probs = classifier.predict_proba([[clock /720, score /53]])
		result = probs[0][1] * 100


		resp = make_response(str(result)+'%')
		resp.headers['Content-Type'] = "application/json"
		return resp

		return render_template('index.html', message='')
	return render_template('index.html', message='')


@predict_script.route("/", methods=['GET', 'POST'])
@predict_script.route('/getGameIds', methods=['GET', 'POST'])
def getGameIds():
	message = None
	files = os.listdir("games/")

	if request.method == 'POST':
		files = json.dumps(files)
		resp = make_response(files)
		resp.headers['Content-Type'] = "application/json"
		return resp

		return render_template('index.html', message='')
	return render_template('index.html', message='')

@predict_script.route("/", methods=['GET', 'POST'])
@predict_script.route('/getProbsGame', methods=['GET', 'POST'])
def getProbsGame():
	message = None

	if request.method == 'POST':
		file = request.form['myfile']
		data = pd.read_csv("games/" + file)
		with open('classifier_score_clutch_adj_2_KNN_500.pkl',  'rb') as f:
			classifier = pickle.load(f)

		predictions = []
		for i in range(0, len(data)):
			row = data.iloc[i]
			next_row = data.iloc[i+1] if (i < len(data)-1) else data.iloc[i]

			if str(row.score) != 'nan':
				home, away = classifier3.getScores(row.score) 
			else:
				pass

			if "Violation" in row.event_type or "Substitution" in row.event_type or "Ejection" in row.event_type:
				pass # Keep event the same as previous
			else:
				event = classifier3.getEvent(row, next_row)

			time = classifier3.getTime(row.play_clock)
			event_adj = event*classifier3.clutchAdj(time)
			score = home - away + event_adj
			prob = classifier.predict_proba([[time  /720, score /53]])
			predictions.append([row.play_clock, home, away, str(row.home_description), str(row.away_description), prob[0][1]])


		preds = json.dumps(predictions)
		resp = make_response(preds)
		resp.headers['Content-Type'] = "application/json"
		return resp

		return render_template('index.html', message='')
	return render_template('index.html', message='')





if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5000))
	predict_script.run(host='0.0.0.0', port=port)