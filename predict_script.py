# Chris Vrabel
# 5/27/17
# Prediting Game Probabilities

# This python script uses flask for communicate with our javascript.  

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

#Our function for best fit line 
def func(x, a, b, c, d, e, f):
	return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

#Predict the probability using best fit curves
def pred(time, score):
	with open('pred_curves_5degree.pkl', 'rb') as f:
		curves = pickle.load(f)

	#Different to handle negatives and positives
	if score > 0:
		ceil = math.ceil(score)
		floor = math.floor(score)
		if floor == 0:
			remain = score
		else:
			remain = score % floor
	elif score < 0:
		ceil = math.floor(score)
		floor = math.ceil(score)
		if floor == 0:
			remain = score*-1
		else:
			remain = (score%floor)*-1
	else:
		ceil = floor = score

	indexHelp = 60
	pred = 0
	if ceil == floor:
		pred =  func(time, *curves[int(score)+indexHelp])
	else:
		print(remain)
		pred = (remain*func(time, *curves[ceil+indexHelp]) + (1-remain)*func(time, *curves[floor+indexHelp])) 

	# If end of game, force prob to 1 or 0
	# If not end of game, don't allow prob of 1 or 0s
	if score >= 1 and time == 0:
		pred = 1
	elif score <= -1 and time==0:
		pred=0
	elif pred >= 1 and time>0 and score<10:
		pred = 0.9999
	elif pred <= 0 and time>0 and score>-10:
		pred = 0.0001
	elif pred > 1:
		pred = 1
	elif pred < 0:
		pred = 0.0001

	return pred



# Perform individual predictions for the top part of our page
@predict_script.route("/", methods=['GET', 'POST'])
@predict_script.route('/predict', methods=['GET', 'POST'])
def predict():
	message = None
	# with open('classifier_ot_2.pkl', 'rb') as f:
	# 	classifier = pickle.load(f)

	if request.method == 'POST':
		clock = float(request.form['myclock'])
		home = float(request.form['myhome'])
		away = float(request.form['myaway'])
		evt = request.form['myevent']
		
		event = eval("Action."+evt+".value")
		event = event * classifier3.clutchAdj(clock)
		print(event)
		print(home-away)
		score = home - away + event
		# probs = classifier.predict_proba([[clock /720, score /53]])
		# result = probs[0][1] * 100

		probs = pred(clock, score)
		result = probs * 100


		resp = make_response(str(result)+'%')
		resp.headers['Content-Type'] = "application/json"
		return resp

		return render_template('index.html', message='')
	return render_template('index.html', message='')


# Get the name of each game csv for our dropdown menu
@predict_script.route("/", methods=['GET', 'POST'])
@predict_script.route('/getGameIds', methods=['GET', 'POST'])
def getGameIds():
	message = None
	# files = os.listdir("games/")
	# files.sort()
	with open('game_names.pkl', 'rb') as f:
		games = pickle.load(f)


	if request.method == 'POST':
		files = json.dumps(games)
		resp = make_response(files)
		resp.headers['Content-Type'] = "application/json"
		return resp

		return render_template('index.html', message='')
	return render_template('index.html', message='')

# Used for the bottom of our page. Get all values within a specified probability window
@predict_script.route("/", methods=['GET', 'POST'])
@predict_script.route('/getProbWindow', methods=['GET', 'POST'])
def getProbWindow():

	# with open('classifier_ot_2.pkl', 'rb') as f:
	# 	classifier = pickle.load(f)
	with open('pred_curves.pkl', 'rb') as f:
		curves = pickle.load(f)

	if request.method == 'POST':
		lower = float(request.form['mylower'])
		upper = float(request.form['myupper'])

		values = []
		for t in range(720,-1,-10):
			for s in range(-24, 25, 1):
				# probs = classifier.predict_proba([[t  /720, s /53]])
				# if probs[0][1] <= upper and probs[0][1] >= lower:
				# 	values.append([t,s,probs[0][1]])

				probs = pred(t, s)
				# probs = func(t, *curves[s+55])
				if probs <= upper and probs >= lower:
					values.append([t,s,probs])

		# values = []
		# for t in range(721,-1,-1):
		# 	probs = classifier.predict_proba([[t  /720, 1 /53]])
		# 	values.append([t,probs[0,1]])


		values = json.dumps(values)
		resp = make_response(values)
		resp.headers['Content-Type'] = "application/json"
		return resp

		return render_template('index.html', message='')
	return render_template('index.html', message='')


# Get probabilities for each event throughout a specified game
@predict_script.route("/", methods=['GET', 'POST'])
@predict_script.route('/getProbsGame', methods=['GET', 'POST'])
def getProbsGame():
	message = None

	if request.method == 'POST':
		file = request.form['myfile']
		data = pd.read_csv("games/" + file)
		# with open('classifier_ot_2.pkl',  'rb') as f:
		# 	classifier = pickle.load(f)

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
			period = int(row.period)
			# if(period > 4):
			# 	time = time - 300
			event_adj = event*classifier3.clutchAdj(time)
			score = home - away + event_adj
			# prob = classifier.predict_proba([[time  /720, score /53]])
			# predictions.append([row.play_clock, home, away, str(row.home_description), str(row.away_description), prob[0][1], int(row.period)])
			print(score)
			probs = pred(time, score)
			predictions.append([row.play_clock, home, away, str(row.home_description), str(row.away_description), probs, int(row.period)])

		preds = json.dumps(predictions)
		resp = make_response(preds)
		resp.headers['Content-Type'] = "application/json"
		return resp

		return render_template('index.html', message='')
	return render_template('index.html', message='')




if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5000))
	predict_script.run(host='0.0.0.0', port=port)