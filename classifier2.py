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
	homeMake = -0.7116
	awayMake = 0.7116
	homeMiss = -0.5444
	awayMiss = 0.5444
	homeTurnover = -0.7116
	awayTurnover = 0.7116
	homeFt1 = 0.5
	homeFt2 = 1.0
	homeFt3 = 1.5
	awayFt1 = -0.5
	awayFt2 = -1.0
	awayFt3 = -1.5
	homeFoulNonS = -0.7116
	awayFoulNonS = 0.7116
	homeFreeMakeLast = -0.7116
	awayFreeMakeLast = 0.7116
	homeFreeMiss = -0.5444
	awayFreeMiss = 0.5444
	homeRebound = 0.7116
	awayRebound = -0.7116
	homeJumpBall = 0.7116
	awayJumpBall = -0.7116
	homeTimeout = 0.7116
	awayTimeout = -0.7116


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
		return Action.homeMake.value
	else:
		return Action.awayMake.value

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
	if str(row.home_description) != 'nan':
		if 'of 1' in str(next_row.event_description):
			return Action.awayFt1.value
		elif 'of 2' in str(next_row.event_description):
			return Action.awayFt2.value
		elif 'of 3' in str(next_row.event_description):
			return Action.awayFt3.value
		elif 'Shooting' in str(row.event_description):
			return Action.awayFt2.value
		elif 'Technical' in str(row.event_description):
			return Action.awayFt1.value
		else:
			return Action.homeFoulNonS.value

	elif str(row.away_description) != 'nan':
		if 'of 1' in str(next_row.event_description):
			return Action.homeFt1.value
		elif 'of 2' in str(next_row.event_description):
			return Action.homeFt2.value
		elif 'of 3' in str(next_row.event_description):
			return Action.homeFt3.value
		elif 'Shooting' in str(row.event_description):
			return Action.homeFt2.value
		elif 'Technical' in str(row.event_description):
			return Action.homeFt1.value
		else:
			return Action.awayFoulNonS.value

	elif 'Double Technical' in str(row.event_description):
		return Action.push.value

	elif 'Technical' in str(row.event_description):
		if str(next_row.home_description) != 'nan':
			return Action.homeFt1.value
		elif str(next_row.away_description) != 'nan':
			return Action.awayFt1.value
		else:
			return Action.push.value

	elif 'Double Personal' in str(row.event_description):
		return Action.push.value

def getFreeThrowType(row):
	if str(row.home_description) != 'nan':
		if '1 of 2' in str(row.home_description) or '2 of 3' in str(row.home_description):
			return Action.homeFt1.value
		if '1 of 3' in str(row.home_description):
			return Action.homeFt2.value
		elif 'MISS' in str(row.home_description):
			return Action.homeMiss.value
		else:
			return Action.homeMake.value
	else:
		if '1 of 2' in str(row.away_description) or '2 of 3' in str(row.away_description):
			return Action.awayFt1.value
		if '1 of 3' in str(row.away_description):
			return Action.awayFt2.value
		elif 'MISS' in str(row.away_description):
			return Action.awayMiss.value
		else:
			return Action.awayMake.value

def getReboundType(row):
	if 'REBOUND' in str(row.home_description) or 'Rebound' in str(row.home_description):
		return Action.homeRebound.value
	elif 'REBOUND' in str(row.away_description) or 'Rebound' in str(row.away_description):
		return Action.awayRebound.value

def getJumpBallType(row):
	if str(row.player3_team) != 'nan':
		if str(row.player3_team) == str(row.player1_team):
			return Action.homeJumpBall.value
		else:
			return Action.awayJumpBall.value
	else:
		return Action.push.value

def getTimeoutType(row):
	if str(row.home_description) != 'nan':
		return Action.homeTimeout.value
	else:
		return Action.awayTimeout.value

def getEnd(row):
	home, away = getScores(row.score)
	if home > away:
		return 100
	elif away > home:
		return -100
	else:
		return 0

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
		return getEnd(row)

	elif "Violation" in row.event_type:
		return Action.push.value

	elif "Timeout" in row.event_type:
		return getTimeoutType(row)

	elif "Instant" in row.event_type:
		return Action.push.value

	elif "Ejection" in row.event_type:
		return Action.push.value

	return Action.push.value

def parseCSV(filename):
	x = []
	y = []
	data = pd.read_csv(filename)

	home_score = 0
	away_score = 0
	diff_score = 0
	max_diff = 0
	event = -1
	for i in range(0, len(data)):
		row = data.irow(i)
		next_row = data.irow(i+1) if (i < len(data)-1) else data.irow(i)
		time = getTime(row.play_clock)


		if str(row.score) != 'nan':
			home_score, away_score = getScores(row.score) 
			diff_score = home_score - away_score
		else:
			pass
		if "Violation" in row.event_type or "Substitution" in row.event_type or "Ejection" in row.event_type:
			pass # Keep event the same as previous
		else:
			event = getEvent(row, next_row)

		Xtemp = [time/720, (home_score-away_score)/53, event]
		x.append(Xtemp)

		if event is None:
			print(row.sequence_id)
			print(event)


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


	with open('Xvector_poss_event.pkl', 'wb') as f:
		pickle.dump(Xvector, f)

	with open('Yvector_poss_event.pkl', 'wb') as f:
		pickle.dump(Yvector, f)



main()