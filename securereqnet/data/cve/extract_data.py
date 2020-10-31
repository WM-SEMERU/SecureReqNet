
import csv
import random

def get_cve_dataset():
	filename = 'cve_dataset.tsv'
	data = []
	with open(filename, 'r') as tsv_file:
		tsv_reader = csv.reader(tsv_file, delimiter='\t')
		for line in tsv_reader:
			data.append((line[1], line[2]))
	return data

def get_test_and_training(dataset, test_ratio=0.1):
	train = list(dataset)
	n_test = int(len(dataset) * test_ratio)

	test = random.sample(dataset, n_test)
	for i in range(n_test):
		train.remove(test[i])

	return test, train

if __name__ == '__main__':
	dataset = get_cve_dataset()
	test, train = get_test_and_training(dataset)

	print(test[0])
