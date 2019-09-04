import os

issue_count = 0

directory = './organizations'
for group in os.listdir(directory):
	group_folder = os.path.join(directory, group)
	if os.path.isdir(group_folder):
		for repo in os.listdir(group_folder):
			repo_folder = os.path.join(group_folder, repo)
			if os.path.isdir(repo_folder):
				issue_count += len(os.listdir(repo_folder)) - 1

print("Issue count: {}".format(issue_count))
