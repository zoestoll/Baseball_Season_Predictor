import MapReduce
import sys
import csv

mr = MapReduce.MapReduce()

def mapper(record):
	# INPUT:
	# 	line: a line from either salary_features.csv or batting_features.csv or pitching.csv or fielding.csv
	# OUTPUT:
	# 	key: (year, team)
	# 	value: a list of values in the line

	values = record
	print values
	mr.emit_intermediate(values[0], values[1])


def reducer(key, values):
	# INPUT:
	# 	key: (year, team)
	# 	value: a list of values in the line
	# OUTPUT:
	# 	key: (year, team)
	# 	value: (salary, batting features...)

	feature_list = []
	# print len(values)
	# if len(values) == 24:
	# 	# Batting.csv
	# if len(values) == 5:
	# 	# Salaries.csv
	for line in values:
		feature_list.append(line)
	year = key[0]
	print feature_list[0]
	print feature_list[1]
	print "\n"
	combined_dict = {k: v for d in feature_list[0] for k, v in feature_list[1].items()}
	print combined_dict
	mr.emit((key, feature_list))


if __name__ == '__main__':
  # the code loads the json file and executes 
  # the MapReduce query which prints the result to stdout.

	batting_data = csv.reader(open(sys.argv[1], 'rU'))
	batting_reader = csv.reader(batting_data)
	salary_data = csv.reader(open(sys.argv[2], 'rU'))
	salary_reader = csv.reader(salary_data)
	with open("combined_features.csv", "w") as combined:
		writer = csv.writer(combined)
		for line in batting_data:
			writer.writerow(line)
		for line in salary_data:
			writer.writerow(line)

	all_data = csv.reader(open("combined_features.csv", "rU"))


	outputf = open("all_features.csv", "w")
	writer = csv.writer(outputf)
	mr.execute(all_data, mapper, reducer, outputf)











