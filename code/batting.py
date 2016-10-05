import MapReduce
import sys
import csv

# MAP REDUCE FUNCTION UPDATED

"""
Word Count Example in the Simple Python MapReduce Framework
"""

# we create a MapReduce object that is used to pass data 
# between the map function and the reduce function; 
# you won't need to use this object directly.
mr = MapReduce.MapReduce()

# the mapper function tokenizes each document and 
# emits a key-value pair. The key is a word
# as a string and the value is the words

def blank(item):
    if item != '':
        return int(item)
    else:
        return 0

def mapper(record):
    # Input = line from Batting.csv
    # key: team
    # value: list of player id - 0, year - 1, team - 3, games batting - 6, at bats - 7, hits - 8, 2B - 9,
    # 3B - 10, HR - 11, RBI - 12, stolen bases - 13, walks - 15, strikeouts - 16, intentional walks - 17, sac flies - 20
    # ground into double - 21


    year = record[1]
    team = record[3]
    print year
    if int(year) >= 1985:
        player = record[0] # 0
        games_batting = blank(record[6])
        at_bats = blank(record[7]) # 2
        hits = blank(record[8]) # 3
        doubles = blank(record[9]) # 4
        triples = blank(record[10]) # 5
        hr = blank(record[11]) # 6
        rbi = blank(record[12]) # 7
        stolen_bases = blank(record[13]) # 8
        walks = blank(record[15]) # 9
        strikeouts = blank(record[16]) # 10
        intentional_walks = blank(record[17]) # 11
        sac_flies = blank(record[20]) # 12
        ground_double_play = blank(record[21]) # 13                  0       1           2       3       4       5       6   7       8           9       10          11                  12          13
        mr.emit_intermediate((year, team), [player, games_batting, at_bats, hits, doubles, triples, hr, rbi, stolen_bases, walks, strikeouts, intentional_walks, sac_flies, ground_double_play])

# the reducer function sums up the list of occurrence counts 
# and emits a count for word. Since the mapper function emits 
# the integer 1 for each word, each element in the list_of_values 
# is the integer 1. The list of occurrence counts is summed 
# and a (word, total) tuple is emitted where word is a string and total is an integer.

def reducer(key, list_of_values):

    # key: (year, team)
    # value: list of features

    count = 0
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

    for item in list_of_values:
        avg_games_batting += item[1]
        avg_at_bats += item[2]
        avg_hits += item[3]
        avg_doubles += item[4]
        avg_triples += item[5]
        avg_hr += item[6]
        avg_rbi += item[7]
        avg_sb += item[8]
        avg_walks += item[9]
        avg_strikeouts += item[10]
        avg_intentional += item[11]
        avg_sac_flies += item[12]
        avg_double_plays += item[13]
        count += 1

    avg_games_batting = avg_games_batting/float(count)
    avg_at_bats = avg_at_bats/float(count)
    avg_hits = avg_hits/float(count)
    avg_doubles = avg_doubles/float(count)
    avg_triples = avg_triples/float(count)
    avg_hr = avg_hr/float(count)
    avg_rbi = avg_rbi/float(count)
    avg_sb = avg_sb/float(count)
    avg_walks = avg_walks/float(count)
    avg_strikeouts = avg_strikeouts/float(count)
    avg_intentional = avg_intentional/float(count)
    avg_sac_flies = avg_sac_flies/float(count)
    avg_double_plays = avg_double_plays/float(count)

    team_stats = [{"Average games batting":avg_games_batting, "At bats":avg_at_bats, "Hits":avg_hits, "Doubles":avg_doubles, "Triples":avg_triples, "Home runs":avg_hr, "RBI":avg_rbi, "Stolen bases":avg_sb,
    "Walks":avg_walks, "Strikeouts":avg_strikeouts, "Intentional walks":avg_intentional, "Sac flies":avg_sac_flies, "Double plays":avg_double_plays}]

    mr.emit((key, str(team_stats).strip('[]')))


if __name__ == '__main__':
  # the code loads the json file and executes 
  # the MapReduce query which prints the result to stdout.

    data = csv.reader(open(sys.argv[1], 'rU'))
    outputf = open("batting_features.csv", "w")
    writer = csv.writer(outputf)
    mr.execute(data, mapper, reducer, outputf)
