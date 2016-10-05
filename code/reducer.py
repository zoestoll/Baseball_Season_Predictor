from __future__ import division
from mrjob.job import MRJob
import math
import sys
import operator
import csv
import numpy
from scipy.stats.stats import pearsonr   

VIRTUAL_COUNT = 10
PRIOR_CORRELATION = 0.0
THRESHOLD = 0.5
player_stats = {}
team_players = {}
yearly_ranks = {}

class Similarities(MRJob):

	def mapper_0(self, _, line):
		# INPUT:
		#   line: a line from either salary.csv or batting.csv or pitching.csv or fielding.csv
		# OUTPUT:
		#   key: (year, team, rank)
		#   value: a list of values in the line
		line = line.split("\r") # This works
		for record in line:
			record = record.split(",")
			player = ""
			year = 0
			if len(record) == 5:
				# Salaries.csv
				player = record[3]
				year = record[0]
				team = record[1]
				# values = [team] + record[4:] # [team, salary]
			if len(record) == 24:
				# Batting.csv
				player = record[0]
				year = record[1]
				team = record[3]
				# values = record[1:4] + record[6:18]
			if int(year) > 1984:
				key = (year, team)
				values = player
				yield key, player # Key = (year, team), values = (player from (year, team))


	def reducer_0(self, key, values):
		# INPUT:
		# 	key: (year, team)
		# 	value: list of players for (year, team)
		# OUTPUT:
		# 	key: rank
		# 	value: list of players
		year = key[0]
		team = key[1]
		players = []
		for item in values:
			player = item
			players.append(item)

		rank = yearly_ranks[(year, team)]
		key = (rank, year, team)
		yield key, players


	def reducer_1(self, key, values):
		# INPUT:
		# 	key: (rank, year, team)
		# 	value: list of players for (year, team)
		# OUTPUT:
		# 	key: rank for (year, team)
		# 	value: list of players(year, team) stats for (year-1); if rookie, ignore
		rank = key[0]
		year = key[1]
		team = key[2]
		players = []

		for item in values: # item = [[x], [y]...]
			for i in item:
				players.append(i)

		for player in players:
			if (str(int(year)-1), player) in player_stats:
				key = (rank, year, team)
				values = player_stats[(str(int(year)-1), player)]
				yield key, values

	def reducer_2(self, key, values):
		# INPUT:
		# 	key: (rank, year, team)
		# 	value: list of player stats from season before
		# OUTPUT:
		# 	key: rank
		# 	value: average stat

		rank = key[0]
		year = key[1]
		team = key[2]

		data_count = 1 # Avoid dividing by 0
		pitcher_count = 1
		avg_salary = 0

		# batting features 
		avg_games_batting = 0
		avg_at_bats = 0
		avg_hits = 0
		avg_doubles = 0
		avg_triples = 0
		avg_hr = 0
		avg_rbi = 0
		avg_sb = 0
		avg_walks = 0
		avg_strikeouts = 0
		avg_intentional = 0
		avg_sac_flies = 0
		avg_double_plays = 0

		# pitching features
		gp = 0
		gs = 0
		walks = 0
		strikeouts = 0
		era = 0
		hits = 0

		val_list = []
		for val in values:
			val_list.append(val)

		for item in val_list:
			if len(item) == 14: # batter
				avg_salary += int(item[0])
				avg_games_batting += int(item[1])
				avg_at_bats += int(item[2])
				avg_hits += int(item[3])
				avg_doubles += int(item[4])
				avg_triples += int(item[5])
				avg_hr += int(item[6])
				avg_rbi += int(item[7])
				avg_sb += int(item[8])
				avg_walks += int(item[9])
				avg_strikeouts += int(item[10])
				avg_intentional += int(item[11])
				avg_sac_flies += int(item[12])
				avg_double_plays += int(item[13])
				data_count += 1

			if len(item) == 20: # a pitcher
				avg_salary += int(item[0])
				gp += int(item[14])
				gs += int(item[15])
				walks += int(item[16])
				strikeouts += int(item[17])
				era += float(item[18])
				hits += int(item[19])
				pitcher_count += 1
				
		pitcher_count = float(pitcher_count)
		data_count = float(data_count)

		gp = gp/pitcher_count
		gs = gs/pitcher_count
		walks = walks/pitcher_count
		strikeouts = strikeouts/pitcher_count
		era = era/pitcher_count
		hits = hits/pitcher_count

		avg_salary = avg_salary/data_count
		avg_games_batting = avg_games_batting/data_count
		avg_at_bats = avg_at_bats/data_count
		avg_hits = avg_hits/data_count
		avg_doubles = avg_doubles/data_count
		avg_triples = avg_triples/data_count
		avg_hr = avg_hr/data_count
		avg_rbi = avg_rbi/data_count
		avg_sb = avg_sb/data_count
		avg_walks = avg_walks/data_count
		avg_strikeouts = avg_strikeouts/data_count
		avg_intentional = avg_intentional/data_count
		avg_sac_flies = avg_sac_flies/data_count
		avg_double_plays = avg_double_plays/data_count

		avgs_list = [avg_salary, avg_games_batting, avg_at_bats, avg_hits, avg_doubles, avg_triples, avg_hr, avg_rbi, avg_sb, avg_walks,
						avg_strikeouts, avg_intentional, avg_sac_flies, avg_double_plays, gp, gs, walks, strikeouts, era, hits]

		yield(rank, avgs_list)


	def reducer_3(self, key, values):
		# Input: key = rank, values = list of player statistical averages for each year corresponding to this rank.
		# Output: key = feature, values = average of all these averages for the particular feature

		list_of_batter_features = ["Salary", "Games batting", "At bats", "Hits", "Doubles", "Triples",
							"Home runs", "RBI", "Stolen bases", "Walks", "Strike outs", "Intentional walks", "Sac flies", "Double plays"]
		list_of_pitcher_features = ["Pitcher Walks", "Pitcher Strikeouts", "ERA", "Pitcher Hits"]
		vital_features = ["Salary", "Hits", "Home runs", "RBI", "Strike outs", "Pitcher Walks", "Pitcher Strikeouts", "ERA", "Pitcher Hits"]

		rank = key
		avgs_list = []

		avg_salary = 0

		# Batting averages
		gb_avg = 0
		ab_avg = 0
		h_avg = 0
		d_avg = 0
		t_avg = 0
		b_hr_avg = 0
		rbi_avg = 0
		sb_avg = 0
		b_walks_avg = 0
		b_so_avg = 0
		intentional_avg = 0
		sf_avg = 0
		dp_avg = 0

		# Pitching averages
		gp_avg = 0
		gs_avg = 0
		p_walks_avg = 0
		p_so_avg = 0
		era_avg = 0
		p_hits_avg = 0

		data_count = 1

		for feat in values: # all of length 20

			data_count += 1
			salary = feat[0]
			gb = feat[1]
			ab = feat[2]
			ah = feat[3]
			ad = feat[4]
			at = feat[5]
			ahr = feat[6]
			rbi = feat[7]
			sb = feat[8]
			batter_walks = feat[9]
			batter_so = feat[10]
			intentional = feat[11]
			sf = feat[12]
			dp = feat[13]

			gp = feat[14]
			gs = feat[15]
			pitcher_walks = feat[16]
			pitcher_so = feat[17]
			era = feat[18]
			pitcher_hits = feat[19]

			avg_salary += salary
			gb_avg += gb
			ab_avg += ab
			h_avg += ah
			d_avg += ad
			t_avg += at
			b_hr_avg += ahr
			rbi_avg += rbi
			sb_avg += sb
			b_walks_avg += batter_walks
			b_so_avg += batter_so
			intentional_avg += intentional
			sf_avg += sf
			dp_avg += dp

			gp_avg += gp
			gs_avg += gs
			p_walks_avg += pitcher_walks
			p_so_avg += pitcher_so
			era_avg += era
			p_hits_avg += pitcher_hits

		data_count = float(data_count)

		avg_salary = avg_salary/data_count
		gb_avg = gb_avg/data_count
		ab_avg = ab_avg/data_count
		h_avg = h_avg/data_count
		d_avg = d_avg/data_count
		t_avg = t_avg/data_count
		b_hr_avg = b_hr_avg/data_count
		rbi_avg = rbi_avg/data_count
		sb_avg = sb_avg/data_count
		b_walks_avg = b_walks_avg/data_count
		b_so_avg = b_so_avg/data_count
		intentional_avg = intentional_avg/data_count
		sf_avg = sf_avg/data_count
		dp_avg = dp_avg/data_count

		gp_avg = gp_avg/data_count
		gs_avg = gs_avg/data_count
		p_walks_avg = p_walks_avg/data_count
		p_so_avg = p_so_avg/data_count
		era_avg = era_avg/data_count
		p_hits_avg = p_hits_avg/data_count

		avgs_list = [avg_salary, gb_avg, ab_avg, h_avg, d_avg, t_avg, b_hr_avg, rbi_avg, sb_avg, b_walks_avg, b_so_avg, intentional_avg, sf_avg,
					dp_avg, gp_avg, gs_avg, p_walks_avg, p_so_avg, era_avg, p_hits_avg]



		vital_avgs_list = {"Salary": avg_salary, "Hits": h_avg, "Home runs": b_hr_avg, "RBI": rbi_avg, "Strike outs": b_so_avg, "Pitcher Walks": p_walks_avg, "Pitcher Strikeouts": p_so_avg, "ERA": era_avg, "Pitcher Hits": p_hits_avg}
		
		feature_count = 0
		for feature in list_of_batter_features:
			key = list_of_batter_features[feature_count]
			avg_stat = avgs_list[feature_count]
			values = (rank, data_count, [avg_stat]) # Rank, number of stats, and average stat
			feature_count += 1
			yield key, values

		feature_count = 0 # starting at pitcher values
		for feature in list_of_pitcher_features:
			key = list_of_pitcher_features[feature_count]
			avg_stat = avgs_list[feature_count + 14]
			values = (rank, data_count, [avg_stat]) # Rank, number of stats, and average stat
			feature_count += 1


		feature_combinations_2 = combinations(vital_features, 2)
		vc_list = []
		for item in feature_combinations_2:
			vc_list.append(item)

		for vf_combo in vc_list:
			vf_1 = vf_combo[0]
			vf_2 = vf_combo[1]
			avg_1 = vital_avgs_list[vf_1] # Need to be normalized before being sent to next reducer
			# bc otherwise values that are absolutely larger will be treated as being relatively larger as well
			avg_2 = vital_avgs_list[vf_2]
			combined_avg = [int(avg_1), int(avg_2)]
			key = vf_1 + ", " + vf_2
			values = (rank, data_count, combined_avg)
			yield key, values



	def reducer_4(self, key, values):
		# INPUT:
		# 	key: feature type
		# 	value: Rank, number of stats, and average stat for a SINGLE PLAYER
		# OUTPUT
		# 	key:
		# 	value:
		feature = key

		sum_xx = 0
		sum_yy = 0
		sum_xy = 0
		sum_y = 0
		sum_x = 0
		n = 0
		n_x = 0
		n_y = 0

		stat_rank = {}
		combo_stat_rank_1 = {}
		combo_stat_rank_2 = {}

		vals = []
		combo = 0
		for item in values:
			vals.append(item)
			rank = int(item[0]) # RANK
			stat = item[2] # STATISTIC
			if len(stat) == 1: # Not a combination
				stat_rank[rank] = int(stat[0]) # Stat rank: {Actual rank: statistic}
				combo = 1
			if len(stat) == 2: # A combination of two features

				stat_1 = int(stat[0])
				stat_2 = int(stat[1])

				combo_stat_rank_1[rank] = stat_1
				combo_stat_rank_2[rank] = stat_2

				combo = 2

		if combo == 1: # Stat rank: {Actual rank: statistic}
			sorted_stat_ranks = sorted(stat_rank.items(), key=operator.itemgetter(1))
			# Sorted stat rank: [(Actual rank: statistic)] sorted by statistic
			normalized_stat_rank = {}
			statr = 1
			for tup in sorted_stat_ranks:
				rank = tup[0]
				stat = tup[1]
				normalized_stat_rank[rank-1] = statr # Normalized: {Rank-1: Statistic}
				statr += 1

		if combo == 2: # Combination of features

			sorted_stat_ranks_1 = sorted(combo_stat_rank_1.items(), key=operator.itemgetter(1), reverse=False) # sorting by the statistical value
			sorted_stat_ranks_2 = sorted(combo_stat_rank_2.items(), key=operator.itemgetter(1), reverse=False) # sorting by the statistical value

			combined_ranks_1 = {}
			combined_ranks_2 = {}
			combined_ranks = {}

			statr = 1
			for tup in sorted_stat_ranks_1:
				rank = tup[0]
				combined_ranks_1[rank-1] = statr
				statr += 1

			statr = 1
			for tup in sorted_stat_ranks_2:
				rank = tup[0]
				combined_ranks_2[rank-1] = statr
				statr += 1

			for i in range(0, len(combined_ranks_1)):
				comb = combined_ranks_1[i] + combined_ranks_2[i]
				combined_ranks[i] = comb

			normalized_stat_rank = combined_ranks
			# Normalized: {Actual rank: combined rank for both features}


		a = []
		b = []
		for item in vals:
			n += 1

			x = int(item[0]) # RANK
			y = int(normalized_stat_rank[x-1]) # STATISTIC RANK

			a.append(x)
			b.append(y)

			sum_x += x
			sum_xx += x * x

			sum_y += y
			sum_yy += y * y
			sum_xy += x * y
			num_rankings = item[1]

		n_x = num_rankings
		n_y = num_rankings
		correlation_value = correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy)
		regularized_correlation_value = regularized_correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy, VIRTUAL_COUNT, PRIOR_CORRELATION)
		if abs(regularized_correlation_value) > 0.2:
			print feature, ",", correlation_value

	def steps(self):
		return [
			self.mr(mapper=self.mapper_0,
					reducer=self.reducer_0),
			self.mr(reducer=self.reducer_1),
			self.mr(reducer=self.reducer_2),
			self.mr(reducer=self.reducer_3),
			self.mr(reducer=self.reducer_4)
		]

##### Helper Functions ############################################################################

def find_ranks():
	# Input: None
	# Purpose: Makes the global dictionary for the team's ranks per season.
	# Output: None
 
	team_file = open('/home/zstoll/course/cs1951-A/final_project/final/code/Teams.csv', "r")
	teams_reader = csv.reader(team_file)
	next(teams_reader, None)
	team_averages = {}
	for line in teams_reader:
		year = line[0]
		if int(year) < 1985:
			continue
		team = line[2]
		wins = int(line[8])
		losses = int(line[9])
		total_games = wins + losses
		team_avg = wins/float(total_games) # average number of games won
		if year in team_averages:
			team_averages[year].append((team, team_avg))
		else:
			team_averages[year] = [(team, team_avg)]
	for k, v in team_averages.iteritems(): # Once per year
		# Ranks: {year: [(team, standing), (team, standing)...]}
		sorted_averages = sorted(v, key=lambda x: v[1], reverse=False)
		rank = 0
		for team_info in sorted_averages: # 2009 - WAS - 25 , once per team
			year = k
			rank += 1
			team = team_info[0]
			yearly_ranks[(year, team)] = rank # should be 25

def teams():
	# Input: None
	# Purpose: fills in the global player_stats dictionary, which allows us to easily access the stats
	#  for any player for a particular season. This was necessary through out the mapreduce function
	#  which is why I decided to create this global dictionary. It makes it very convenient when fetching
	#  player stats.
	#
	# I would have ideally also factored the Fielding stats in here as well, although that would take a lot more
	#  time and effort to do. As I continue to develop on this project in the future, however, I plan to include Fielding.csv
	#  since that data may also prove vital in determining the final ranking for a team.
	#
	# Output: None

	# Salaries
	with open('/home/zstoll/course/cs1951-A/final_project/final/code/Salaries.csv', "rU") as salary_file:
		salary_reader = csv.reader(salary_file)
		next(salary_reader, None)
		for line in salary_reader:
			year = line[0]
			team = line[1]
			player = line[3]
			salary = line[4]
			if (year, player) in player_stats:
				player_stats[(year, player)].append(salary)
			else:
				player_stats[(year, player)] = [salary]

	# Batting

	with open('/home/zstoll/course/cs1951-A/final_project/final/code/Batting.csv', "rU") as batting_file:
		batting_reader = csv.reader(batting_file)
		next(batting_reader, None)

		for record in batting_reader:
			year = record[1]
			if int(year) > 1984:
				team = record[3]
				player = record[0]
				games_batting = blank(record[6])
				at_bats = blank(record[7]) # 2
				hits = blank(record[9]) # 3
				doubles = blank(record[10]) # 4
				triples = blank(record[11]) # 5
				hr = blank(record[12]) # 6
				rbi = blank(record[13]) # 7
				stolen_bases = blank(record[14]) # 8
				walks = blank(record[16]) # 9
				strikeouts = blank(record[17]) # 10
				intentional_walks = blank(record[18]) # 11
				sac_flies = blank(record[22]) # 12
				ground_double_play = blank(record[23]) # 13 

				if (year, player) in player_stats and len(player_stats[(year, player)]) == 14:
					# For players who switched teams during the season
					player_stats[(year, player)][1] += games_batting
					player_stats[(year, player)][2] += at_bats
					player_stats[(year, player)][3] += hits
					player_stats[(year, player)][4] += doubles
					player_stats[(year, player)][5] += triples
					player_stats[(year, player)][6] += hr
					player_stats[(year, player)][7] += rbi
					player_stats[(year, player)][8] += stolen_bases
					player_stats[(year, player)][9] += walks
					player_stats[(year, player)][10] += strikeouts
					player_stats[(year, player)][11] += intentional_walks
					player_stats[(year, player)][12] += sac_flies
					player_stats[(year, player)][13] += ground_double_play
				elif (year, player) in player_stats and len(player_stats[(year, player)]) == 1:
					# Only played on one team
					player_stats[(year, player)] = player_stats[(year, player)] + [games_batting, at_bats, hits, doubles, triples, hr, rbi, stolen_bases,
												walks, strikeouts, intentional_walks, sac_flies, ground_double_play]

	with open('/home/zstoll/course/cs1951-A/final_project/final/code/Pitching.csv', "rU") as pitching_file:
		pitching_reader = csv.reader(pitching_file)
		next(pitching_reader, None)
		for line in pitching_reader:
			player = line[0]
			year = line[1]
			team = line[3]
			wins = blank(line[5])
			games_pitched = blank(line[6])
			if games_pitched < 5:
				continue
			games_started = blank(line[7])
			losses = blank(line[6])
			walks = blank(line[16])
			strikeouts = blank(line[17])
			era = line[19]
			hits = blank(line[13])
			if (year, player) in player_stats:
				player_stats[(year, player)] = player_stats[(year, player)] + [games_pitched, games_started, walks, strikeouts, era, hits]
			else:
				player_stats[(year, player)] = [games_pitched, games_started, walks, strikeouts, era, hits]

def blank(item):
	if item != '':
		return int(item)
	else:
		return 0

def normalizer(non_normalized_list):

	stat_rank = {}
	vals = []
	for item in non_normalized_list:
		vals.append(item)
		rank = int(item[0]) # RANK
		stat = int(item[2]) # STATISTIC
		stat_rank[rank] = stat

	sorted_stat_ranks = sorted(stat_rank.items(), key=operator.itemgetter(1))

	normalized_stat_rank = {}
	statr = 1
	for tup in sorted_stat_ranks:
		rank = tup[0]
		stat = tup[1]
		normalized_stat_rank[rank-1] = statr
		statr += 1
	# print "noramlized stat ranks: ", normalized_stat_rank
	return vals, normalized_stat_rank

##### Metric Functions ############################################################################
def correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy):
	# http://en.wikipedia.org/wiki/Correlation_and_dependence
	numerator = n * sum_xy - sum_x * sum_y
	denominator = math.sqrt(n * sum_xx - sum_x * sum_x) * math.sqrt(n * sum_yy - sum_y * sum_y)
	if denominator == 0:
		return 0.0
	return numerator / denominator

def regularized_correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy, virtual_count, prior_correlation):
	unregularized_correlation_value = correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy)
	weight = n / (n + virtual_count)
	return weight * unregularized_correlation_value + (1 - weight) * prior_correlation

#####################################################################################################

##### util ##########################################################################################
def combinations(iterable, r):
	# http://docs.python.org/2/library/itertools.html#itertools.combinations
	# combinations('ABCD', 2) --> AB AC AD BC BD CD
	# combinations(range(4), 3) --> 012 013 023 123
	pool = tuple(iterable)
	n = len(pool)
	if r > n:
		return
	indices = range(r)
	yield tuple(pool[i] for i in indices)
	while True:
		for i in reversed(range(r)):
			if indices[i] != i + n - r:
				break
		else:
			return
		indices[i] += 1
		for j in range(i+1, r):
			indices[j] = indices[j-1] + 1
		yield tuple(pool[i] for i in indices)
#####################################################################################################

if __name__ == '__main__':
	find_ranks()
	teams()
	Similarities.run()

