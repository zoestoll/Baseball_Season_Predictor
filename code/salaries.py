import MapReduce
import sys
import csv

# MAP REDUCE FUNCTION UPDATED

"""
Salary in the Simple Python MapReduce Framework
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
    # Input = line from Salaries.csv
    # key: team
    # value: yearID, teamID, salary

    year = record[0]
    team = record[1]
    salary = record[4]

    mr.emit_intermediate((year,team), salary)

# the reducer function sums up the list of occurrence counts 
# and emits a count for word. Since the mapper function emits 
# the integer 1 for each word, each element in the list_of_values 
# is the integer 1. The list of occurrence counts is summed 
# and a (word, total) tuple is emitted where word is a string and total is an integer.

def reducer(key, salaries):
    # key: (year, team)
    # value: salaries
    total = len(salaries)
    sum_salaries = 0
    for item in salaries:
        sum_salaries += int(item)
    avg_salary = sum_salaries/float(total)
    salary_dict = {"Average salary":avg_salary}
    print salary_dict

    mr.emit((key, salary_dict))

if __name__ == '__main__':
  # the code loads the json file and executes 
  # the MapReduce query which prints the result to stdout.
  data = csv.reader(open(sys.argv[1], 'rU'))
  outputf = open("salary_features.csv", "w")
  mr.execute(data, mapper, reducer, outputf)
