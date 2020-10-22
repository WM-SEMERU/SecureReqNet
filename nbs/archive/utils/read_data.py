import random
import sys
import os.path
import zipfile
from pathlib import Path
import pandas as pd

"@danaderp May'20 Refactoring for enhancing time complexity with pandas vectorization"

class Dynamic_Dataset:
	"""
	This class efficiently 'stores' a dataset. Only a list of filenames and
	mappings to their ground truth values are stored in memory. The file
	contents are only brought into memory when requested.

	This class supports indexing, slicing, and iteration.

	A user can treat an instance of this class exactly as they would a list.
	Indexing an instance of this class will return a tuple consisting of
	the ground truth value and the file content of the filename at that index.

	A user can request the filename at an index with get_id(index)

	Example:

		dataset = Dynamic_Dataset(ground_truth)

		print(dataset.get_id(0))
			-> gitlab_79.txt

		print(dataset[0])
			-> ('(1,0)', 'The currently used Rails version, in the stable ...

		for x in dataset[2:4]:
			print(x)
				-> ('(1,0)', "'In my attempt to add 2 factor authentication ...
				-> ('(1,0)', 'We just had an admin accidentally push to a ...

	"""

	def __init__(self, ground_truth, path, isZip):
		'''
		@param ground_truth (dict): A dictionary mapping filenames to ground truth values
		'''
		self.__keys = list(ground_truth.keys())
		self.__ground_truth = ground_truth
		self.__path = path
		self.__isZip = isZip

	def __get_issue(self, filename):
		if self.__isZip:
			paths = [str(x) for x in Path(self.__path).glob("**/*.zip")]
			for onezipath in paths:
				archive = zipfile.ZipFile( onezipath, 'r')
				contents = archive.read('issues/' + filename)
		else:
			with open(self.__path+'issues/' + filename, 'r') as file:
				contents = file.read()
		return contents.strip()

	def get_id(self, index):
		return self.__keys[index]

	def __len__(self):
		return len(self.__keys)

	def __setitem__(self, key, item):
		raise ValueError

	def __getitem__(self, key):
		if type(key) == slice:
			new_keys = self.__keys[key.start:key.stop:key.step]
			new_gt = dict()
			for key in new_keys:
				new_gt[key] = self.__ground_truth[key]
			return Dynamic_Dataset(new_gt)
		else:
			id = self.__keys[key]
			return (self.__ground_truth[id], self.__get_issue(id))

	def __iter__(self):
		self.__index = 0
		return self

	def __next__(self):
		if self.__index < len(self.__keys):
			to_return = self[self.__index]
			self.__index += 1
			return to_return
		else:
			raise StopIteration

class Processing_Dataset:
    """
    A class to wrap up processing functions 
    """
    
    def __init__(self, path):
        self.__path = path
            
    def get_issue(self, filename):
        with open('combined_dataset/issues/' + filename, 'r') as file:
            contents = file.read()
        return contents.strip()

    def get_ground_truth(self):
        gt = dict()
        #print(sys.path[0])
        #path = "combined_dataset/full_ground_truth.txt"
        #path = os.path.join(sys.path[0], path)
        with open(self.__path+'full_ground_truth.txt') as gt_file:
            for line in gt_file.readlines():
                tokens = line.split()
                filename = tokens[0]
                security_status = tokens[1]
                if filename in gt:
                    raise KeyError("Invalid Ground Truth: Duplicate issue [{}]".format(filename))
                gt[filename] = security_status
        return gt

    def get_test_and_training(self, ground_truth, test_ratio=0.1, isZip = False):
        ids = list(ground_truth.keys())
        sr = []
        nsr = []

        for id in ids:
            if ground_truth[id] == '(1,0)':
                sr.append(id)
            elif ground_truth[id] == '(0,1)':
                nsr.append(id)
            else:
                raise ValueError("There was an issue with ground truth: {} - {}".format(id, ground_truth[id]))


        n_test = int(len(sr) * test_ratio)
        sr_test = random.sample(sr, n_test)
        nsr_test = random.sample(nsr, n_test)

        test_gt = dict()
        train_gt = dict(ground_truth)

        for i in range(n_test):
            sr.remove(sr_test[i])
            test_gt[sr_test[i]] = '(1,0)'
            del train_gt[sr_test[i]]

            nsr.remove(nsr_test[i])
            test_gt[nsr_test[i]] = '(0,1)'
            del train_gt[nsr_test[i]]

        test = Dynamic_Dataset(test_gt,self.__path, isZip)
        train = Dynamic_Dataset(train_gt,self.__path, isZip)

        return (test, train)

if __name__ == '__main__':
	ground_truth = get_ground_truth()
	dataset = Dynamic_Dataset(ground_truth)

	test, train = get_test_and_training(ground_truth)

	print(test[0])
	print(train[0])
