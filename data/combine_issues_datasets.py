
import os
import subprocess

security_count = 0
non_security_count = 0

issues = set()

output_folder = 'combined_dataset/'

if not os.path.exists(output_folder + '/issues/'):
	os.makedirs(output_folder + '/issues/')

full_ground_truth_file = open(output_folder + 'full_ground_truth.txt', 'w+')

# First add gitlab dataset
print("Adding GitLab dataset")
with open('gitlab_dataset/ground_truth.txt', 'r') as gitlab_ground_truth_file:
	for line in gitlab_ground_truth_file.readlines():
		tokens = line.split()
		filename = tokens[0] + '.txt'
		if filename in issues:
			print("Skipping duplicate issue: {}".format(filename))
			continue
		issues.add(filename)
		security_status = tokens[1]
		if security_status == '(1,0)':
			security_count += 1
		elif security_status == '(0,1)':
			non_security_count += 1
		else:
			raise ValueError("Problem with {}!! Unrecognized security status: {}".format(filename, security_status))

		subprocess.call(['cp', 'gitlab_dataset/reqs/{}'.format(filename), '{}issues/{}'.format(output_folder, 'gitlab_{}'.format(filename))])
		full_ground_truth_file.write('gitlab_{} {}\n'.format(filename, security_status))

# Then add github datasets
print("Adding GitHub datasets")
orgs_folder = 'github_datasets/organizations'
for organization in os.listdir(orgs_folder):
	print("Adding {}".format(organization))
	org_folder = os.path.join(orgs_folder, organization)
	for repo in os.listdir(org_folder):
		repo_folder = os.path.join(org_folder, repo)
		if not os.path.isdir(repo_folder):
			continue
		with open(os.path.join(repo_folder, 'ground_truth.txt'), 'r') as repo_gt_file:
			for line in repo_gt_file.readlines():
				tokens = line.split()
				filename = tokens[0] + '.txt'
				if filename in issues:
					print("Skipping duplicate issue: {}".format(filename))
					continue
				issues.add(filename)
				security_status = tokens[1]
				if security_status == '(1,0)':
					security_count += 1
				elif security_status == '(0,1)':
					non_security_count += 1
				else:
					raise ValueError("Problem with {}!! Unrecognized security status: {}".format(filename, security_status))

				subprocess.call(['cp', os.path.join(repo_folder, filename), '{}issues/{}'.format(output_folder, filename)])
				full_ground_truth_file.write('{} {}\n'.format(filename, security_status))

print("Security Related: {}".format(security_count))
print("Non-Security Related: {}".format(non_security_count))
print("Total: {}".format(security_count + non_security_count))
full_ground_truth_file.close()
