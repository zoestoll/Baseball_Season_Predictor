### Team batting averages per year from 1950-2013, compared with with final ranking

### Example output:
# 1950: NYY: 3.10
# 1950: BOS: 2.20
# 1951: OAK: 4.50
# ...

#   0         1      2          3         4       5     6     7     8   9     10   
# yearID    lgID    teamID  franchID    divID   Rank    G   Ghome   W   L   DivWin  

from __future__ import division
from mrjob.job import MRJob
import math
import sys
import operator
import csv
import argparse
import collections

# This function outputs the first file for training to the classifier


def standings(input_file):
    data = open(input_file, "rb")
    reader = csv.reader(data)
    next(reader, None)
    years = {}
    ranks = {}

    for line in reader:
        year = line[0]
        if int(year) < 1985:
            continue
        team = line[2]

        # print "year team here: ", year, team

        wins = int(line[8])
        losses = int(line[9])
        total_games = wins + losses
        team_avg = wins/float(total_games)

        if year in years:
            years[(year, team)].append(team_avg)
        else:
            years[(year, team)] = [team_avg]     
        if year in ranks:
            ranks[year].append((team, team_avg))
        else:
            ranks[year] = [(team, team_avg)]

    output_file = open("standings.csv", "w")
    writer = csv.writer(output_file)

    # for k, v in ranks.iteritems():
    # sorted_ranks = sorted(ranks, key=lambda key: ranks[key])
    # print sorted_ranks
    od = collections.OrderedDict(sorted(ranks.items()))
    print od
    # for k, v in years.iteritems():
    #     writer.writerow([k] + years[k])

    normalized_ranks = {}

    for k, v in ranks.iteritems(): # Once per year
        # Ranks: {year: [(team, standing), (team, standing)...]}
        sorted_ranks = sorted(v, key=lambda x: v[1], reverse=True)
        print "Sorted ranks for: ", k, ": ",  sorted_ranks, "\n"
        rank = 0
        for team_info in sorted_ranks: # 2009 - WAS - 25 , once per team
            year = k
            rank += 1
            team = team_info[0]
            normalized_ranks[(year, team)] = rank # should be 25

    for k, v in normalized_ranks.iteritems():
        print k, v
        
        writer.writerow([k] + [normalized_ranks[k]])
    # Output:
    # (year, team): team average


if __name__ == '__main__':
    ##### DO NOT MODIFY THESE OPTIONS ##########################
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputf', required=True, help='Path to data')
    opts = parser.parse_args()
    ############################################################

    standings(opts.inputf)

