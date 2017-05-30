import pandas as pd
import numpy as np
import os
import math
from sklearn import svm
from enum import Enum
import pickle
from sklearn.model_selection import train_test_split

class Action(Enum):
	push = 0
	homeThree = 1
	awayThree = -1
	homeTwo = 2
	awayTwo = -2
	homeMiss = 3
	awayMiss = -3
	homeTurnover = 4
	awayTurnover = -4
	homeFoulSht = 5
	awayFoulSht = -5
	homeFoulNonS = 6
	awayFoulNonS = -6
	homeFreeMake = 7
	awayFreeMake = -7
	homeFreeMiss = 8
	awayFreeMiss = -8
	homeRebound = 9
	awayRebound = -9
	homeJumpBall = 10
	awayJumpBall = -10
	homeViolation = 11
	awayViolation = -11
	homeTimeout = 12
	awayTimeout = -12
	homeEjection = 13
	awayEjection = -13


def getTime(clock):
	index = clock.index(':')
	minutes = int(clock[0:index])
	seconds = int(clock[index+1:])

	return minutes*60 + seconds

def getScores(score):
	index = score.index('-')
	home = int(score[0 : index-1])
	away = int(score[index+2 : ])
	return home, away

def getShotType(row):
	if(str(row.home_description) != 'nan'):
		if '3PT' in str(row.home_description):
			return Action.homeThree.value
		else:
			return Action.homeTwo.value
	else:
		if '3PT' in str(row.away_description):
			return Action.awayThree.value
		else:
			return Action.awayTwo.value

def getMissShotType(row):
	if(str(row.home_description) != 'nan'):
		return Action.homeMiss.value
	else:
		return Action.awayMiss.value

def getTurnoverType(row):
	if 'Turnover' in str(row.home_description):
		return Action.homeTurnover.value
	elif 'Turnover' in str(row.away_description):
		return Action.awayTurnover.value

def getFoulType(row, next_row):
	if 'FOUL' or 'Foul' in str(row.home_description):
		if 'Free Throw' in str(next_row.event_type):
			return Action.homeFoulSht.value
		else:
			return Action.homeFoulNonS.value
	elif 'FOUL' or 'Foul' in str(row.away_description):
		if 'Free Throw' in str(next_row.event_type):
			return Action.awayFoulSht.value
		else:
			return Action.awayFoulNonS.value

def getFreeThrowType(row):
	if str(row.home_description) != 'nan':
		if 'MISS' in str(row.home_description):
			return Action.homeFreeMiss.value
		else:
			return Action.homeFreeMake.value
	else:
		if 'MISS' in str(row.away_description):
			return Action.awayFreeMiss.value
		else:
			return Action.awayFreeMiss.value

def getReboundType(row):
	if 'REBOUND' or 'Rebound' in str(row.home_description):
		return Action.homeRebound.value
	elif 'REBOUND' or 'Rebound' in str(row.away_description):
		return Action.awayRebound.value

def getJumpBallType(row):
	if str(row.player3_team) != 'nan':
		if str(row.player3_team) == str(row.player1_team):
			return Action.homeJumpBall.value
		else:
			return Action.awayJumpBall.value
	else:
		return Action.push.value

def getViolationType(row):
	if str(row.home_description) != 'nan':
		return Action.homeViolation.value
	else:
		return Action.awayViolation.value

def getTimeoutType(row):
	if str(row.home_description) != 'nan':
		return Action.homeTimeout.value
	else:
		return Action.awayTimeout.value

def getEjectionType(row):
	if str(row.home_description) != 'nan':
		return Action.homeEjection.value
	else:
		return Action.awayEjection.value

def getEvent(row, next_row):
	if "Start Period" in row.event_type:
		return Action.push.value
	elif "Made Shot" in row.event_type:
		return getShotType(row)

	elif "Missed Shot" in row.event_type:
		return getMissShotType(row)

	elif "Turnover" in row.event_type:
		return getTurnoverType(row)

	elif "Foul" in row.event_type:
		return getFoulType(row, next_row)

	elif "Free Throw" in row.event_type:
		return getFreeThrowType(row)

	elif "Rebound" in row.event_type:
		return getReboundType(row)

	elif "Substitution" in row.event_type:
		return Action.push.value

	elif "Jump Ball" in row.event_type:
		return getJumpBallType(row)

	elif "End Period" in row.event_type:
		return Action.push.value

	elif "Violation" in row.event_type:
		return getViolationType(row)

	elif "Timeout" in row.event_type:
		return getTimeoutType(row)

	elif "Instant" in row.event_type:
		return Action.push.value

	elif "Ejection" in row.event_type:
		return getEjectionType(row)

	return Action.push.value

def parseCSV(filename):
	x = []
	y = []
	data = pd.read_csv(filename)

	home_score = 0
	away_score = 0
	diff_score = 0
	max_diff = 0
	for i in range(0, len(data)):
		row = data.irow(i)
		next_row = data.irow(i+1) if (i < len(data)-1) else data.irow(i)
		time = getTime(row.play_clock)
		if str(row.score) != 'nan':
			home_score, away_score = getScores(row.score) 
			diff_score = home_score - away_score
		else:
			pass

		event = getEvent(row, next_row)

		Xtemp = [time/720*50, (home_score-away_score)/53*50, event*3]
		x.append(Xtemp)


	y = [1]*len(data) if home_score-away_score > 0 else [0]*len(data)
	y = np.reshape(y, (-1,1))


	return x, y

def main():

	Xvector = []
	Yvector = []
	for file in os.listdir("games/"):
		print(file)
		
		x, y = parseCSV("games/"+file)
		
		Xvector.extend(x)
		Yvector.extend(y)


	with open('Xvector_normed_scorediff.pkl', 'wb') as f:
		pickle.dump(Xvector, f)

	with open('Yvector_normed_scorediff.pkl', 'wb') as f:
		pickle.dump(Yvector, f)



main()