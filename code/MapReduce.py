# import json
import csv

class MapReduce:
    def __init__(self):
        self.intermediate = {}
        self.result = []

    def emit_intermediate(self, key, value):
        self.intermediate.setdefault(key, [])
        self.intermediate[key].append(value)

    def emit(self, value):
        self.result.append(value) 

    def execute(self, data, mapper, reducer, outputf):
        # inputf = open(data, "rb")
        # reader = csv.reader(data)
        # next(reader, None)
        writer = csv.writer(outputf)
        for line in data:
            mapper(line)

        for key in self.intermediate:
            reducer(key, self.intermediate[key])

        for item in self.result:
            # print item
            writer.writerow(item)
