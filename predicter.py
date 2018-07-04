# Chris Vrabel
# 5/26/17
# Prediting Game Probabilities

# This python script grabs our X and Y vectors and performs a KNN classification.
# REPL can be uncommented to test out classifier.

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


#Our function for best fit line 
def func(x, a, b, c):
	return a*x**2 + b*x + c


def main():
	with open('Xvector_ot_1.pkl', 'rb') as f:
		Xvector = pickle.load(f)

	with open('Yvector_ot_1.pkl', 'rb') as f:
		Yvector = pickle.load(f)

	Yvector = np.array(Yvector)
	Yvector = Yvector.flatten()


	Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xvector, Yvector, test_size=0.1, random_state=42)



	# KNN Classifier
	# --------- Train again -------------- #
	# classifier = KNN(n_neighbors = 500)
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
	# restValues = []
	# for t in range(0,820,5):
	# 	second = False
	# 	for s in range(0, -30, -1):
	# 		probs = classifier.predict_proba([[t  /720, s /53]])
	# 		if probs[0][1] < .015:
	# 			if second == False:
	# 				second = True
	# 			else:
	# 				restValues.append([(720-t),s])
	# 				print(str([t,s]))
	# 				break
	# restValues = np.array(restValues)

	# plt.scatter(restValues[:,0], restValues[:,1], s=120, alpha=0.25, edgecolors='white')
	# popt, pcov = curve_fit(func, restValues[:,0], restValues[:,1])
	# plt.plot(restValues[:,0], func(restValues[:,0], *popt), 'r-')
	# plt.xlabel('4th Quarter Time')
	# plt.ylabel('Point Lead')
	# plt.xticks([0, 120, 240, 360, 480, 600, 720], ['12:00', '10:00', '8:00', '6:00', '4:00', '2:00', '0:00'])
	# plt.xlim([0,720])
	# plt.ylim([-25,0])
	# plt.title('Probability Win < 1.5% Threshold')
	# plt.tight_layout()
	# plt.show()


	#-------------------TESTS-------------------------------------
	# values = []
	# for t in range(0,820,5):
	# 	for s in range(-15, 15, 1):
	# 		probs = classifier.predict_proba([[t  /720, s /53]])
	# 		if probs[0][1] < .85 and probs[0][1] > 0.15:
	# 			values.append([(720-t),s])
	# 			print(str([t,s]))
	# 			# break




	# values = np.array(values)

	# plt.scatter(values[:,0], values[:,1], s=120, alpha=0.25, edgecolors='white')
	# # popt, pcov = curve_fit(func, values[:,0], values[:,1])
	# # plt.plot(values[:,0], func(values[:,0], *popt), 'r-')
	# plt.xlabel('4th Quarter Time')
	# plt.ylabel('Point Lead')
	# plt.xticks([0, 120, 240, 360, 480, 600, 720], ['12:00', '10:00', '8:00', '6:00', '4:00', '2:00', '0:00'])
	# plt.xlim([0,720])
	# plt.ylim([-20,20])
	# plt.title('Close Games')
	# plt.tight_layout()
	# plt.show()







if __name__ == '__main__':
	main()

