import pandas as pd
import numpy as np
import os
import math
from sklearn import svm
from enum import Enum
import pickle
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import classifier3

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


def func(x, a, b, c):
	return a*x**2 + b*x + c


def main():
	with open('Xvector_ot.pkl', 'rb') as f:
		Xvector = pickle.load(f)

	with open('Yvector_ot.pkl', 'rb') as f:
		Yvector = pickle.load(f)

	Yvector = np.array(Yvector)
	Yvector = Yvector.flatten()



	Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xvector, Yvector, test_size=0.1, random_state=42)


	# Support Vector Machine
	# classifier = svm.SVC(C=1,kernel='poly',degree=4, gamma=1, coef0=1, probability=True, max_iter=100)
	# classifier.fit(Xtrain, Ytrain)

	# with open('classifierSVC_poly.pkl', 'wb') as f:
	# 	pickle.dump(classifier, f)







	# KNN Classifier

	# --------- Train again -------------- #
	# classifier = KNN(n_neighbors = 100)
	# classifier.fit(Xtrain, Ytrain)

	# with open('classifier_ot_2.pkl', 'wb') as f:
	# 	pickle.dump(classifier, f, protocol=2)

	# probs = classifier.predict_proba(Xtest)

	# for i in range(0, len(probs)):
	# 	print(str(probs[i]) + " --- " + str(Xtest[i][0]*720) + ", " + str(Xtest[i][1]*53) + " --- " + str(Ytest[i]))
	

	# ---------- Already Trained ------------- #
	with open('classifier_ot_2.pkl', 'rb') as f:
		classifier = pickle.load(f)


	#---------------REPL-------------------------------------
	# while(True):

	# 	timeleft = float(input("Time Left: "))
	# 	homeScore = float(input("Home Score: "))
	# 	awayScore = float(input("Away Score: "))
	# 	evt = input("Last Event: ")
	# 	event = eval("Action."+evt+".value")

	# 	print(event)
	# 	event = event * classifier3.clutchAdj(timeleft)
	# 	score = homeScore - awayScore + event
		
	# 	probs = classifier.predict_proba([[timeleft /720, score /53]])
	# 	print("Probability of winning: " + str(probs[0][1]) + '\n')


	#--------------REST CALCULATIONS-------------------------
	restValues = []
	for t in range(0,820,5):
		for s in range(0, -40, -1):
			# print(s)
			probs = classifier.predict_proba([[t  /720, s /53]])
			if probs[0][1] <= 0.01:
				restValues.append([(720-t),s])
				print(str([t,s]))
				break


	restValues = np.array(restValues)

	plt.scatter(restValues[:,0], restValues[:,1], s=120, alpha=0.25, edgecolors='white')
	popt, pcov = curve_fit(func, restValues[:,0], restValues[:,1])
	plt.plot(restValues[:,0], func(restValues[:,0], *popt), 'r-')
	plt.xlabel('4th Quarter Time')
	plt.ylabel('Point Lead')
	plt.xticks([0, 120, 240, 360, 480, 600, 720], ['12:00', '10:00', '8:00', '6:00', '4:00', '2:00', '0:00'])
	plt.xlim([0,720])
	plt.ylim([-25,0])
	plt.title('Probability Win < 1% Threshold')
	plt.tight_layout()
	plt.show()





def addPredictions(filename):
	df = pd.read_csv("pbp.csv")

	with open('classifier_score_clutch_adj_2_KNN_500.pkl', 'rb') as f:
		classifier = pickle.load(f)

	predictions = []
	prev = 0
	print(prev)
	for i in range(0, len(df)):
		load = int(i/len(df) * 100)
		if load != prev:
			print(load)
			prev = load
		row = df.irow(i)
		next_row = df.irow(i+1) if (i < len(df)-1) else df.irow(i)

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
		if(period > 4):
			time = time - float((period-4)*300)
		event = event*classifier3.clutchAdj(time)
		score = home - away + event
		prob = classifier.predict_proba([[time  /720, score /53]])
		# print(str(home) + "-" + str(away) + " -- " + str(event) + ", " + str(time) + " --- " + str(prob[0][1]))
		predictions.append(prob[0][1])

	df['win_probability'] = predictions

	# df.to_csv(filename, index=False)




if __name__ == '__main__':
	main()

# addPredictions("pbp_predictions.csv")		