
import csv


def get_corona_dataset():
	filename = 'raw_corona.tsv'
	data = []
	with open(filename, 'r') as tsv_file:
		tsv_reader = csv.reader(tsv_file, delimiter='\t')
		for line in tsv_reader:
			if line[-3] == 'SR':
				data.append(('(1,0)', line[3]))
			else:
				data.append(('(0,1)', line[3]))
	return data
