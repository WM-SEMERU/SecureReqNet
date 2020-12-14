import urllib3
from bs4 import BeautifulSoup
import os
import subprocess
import sys

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
http = urllib3.PoolManager()

global_num_issues = 0

group_name = sys.argv[1]
base_url = 'https://github.com/' + group_name

def get_parsed_page(url_requested):
	page_http = http.request('GET', url_requested)
	page_html = str(page_http.data.decode('utf-8'))
	return BeautifulSoup(page_html, 'html.parser')

def get_repo_security_labels(repo_url):
	parsed_labels_html = get_parsed_page(repo_url + '/labels?&q=security')
	return [element.text.strip() for element in parsed_labels_html.find_all("span", attrs={"class": "label-name"})]

def download_issue(repo_name, issue_id, security_related, gt_file, label=None):
	global group_name
	global global_num_issues

	issue_page = get_parsed_page(base_url + '/' + repo_name + '/issues/' + issue_id)

	issue_title = issue_page.find("h1", attrs={"class": "discussion-title"})
	if issue_title is None:
		issue_title = issue_page.find("span", attrs={"class": "js-issue-title"})
	if label is None:
		print("Downloading {} {}".format(repo_name, issue_id))
	else:
		print("Downloading {} {} <{}>".format(repo_name, issue_id, label))
	issue_title = issue_title.text.strip()

	issue_description_element = issue_page.find("div", attrs={"class": "discussion-starting-comment"})
	if issue_description_element is None:
		issue_description_element = issue_page.find("td", attrs={"class": "d-block comment-body markdown-body js-comment-body"})
	issue_description = issue_description_element.text

	global_issue_identifier = '{}_{}_{}'.format(group_name, repo_name, issue_id)
	with open('organizations/{}/{}/{}.txt'.format(group_name, repo_name, global_issue_identifier), 'w+') as issue_file:
		issue_file.write(issue_title + '\n\n')
		issue_file.write(issue_description)
	gt_file.write("{} {}\n".format(global_issue_identifier, '(1,0)' if security_related else '(0,1)'))

	global_num_issues += 1

def get_num_pages(parsed_html):
	pages = parsed_html.find("div", attrs={"class": "pagination"})
	if pages is None:
		return 1
	n_pages = int(pages.findChildren("a")[-2].text)
	return n_pages

def download_issues(repo_name, security_labels):
	global group_name

	if not os.path.exists('organizations/{}/{}'.format(group_name, repo_name)):
		os.makedirs('organizations/{}/{}'.format(group_name, repo_name))

	gt_file = open('organizations/{}/{}/ground_truth.txt'.format(group_name, repo_name), 'w+')
	# First download all issues with security tags
	num_security_issues = 0
	for label in security_labels:
		if any(ord(c) >= 128 for c in label):
			print("{} includes non-ascii characters, ignoring".format(label))
			continue

		needs_quotes = False
		if ' ' in label:
			needs_quotes = True
		url_label = label.replace(' ', '+').replace(':', '%3A')
		if needs_quotes:
			url_label = "\"" + url_label + "\""
		url_label = '%3A' + url_label
		security_issues_html = get_parsed_page(base_url + '/' + repo_name + '/issues?page=1&q=label' + url_label)
		n_pages = get_num_pages(security_issues_html)
		for page in range(1, n_pages + 1):
			issues_page_url = base_url + '/' + repo_name + '/issues?page={}&q=label{}'.format(page, url_label)
			security_issues_html = get_parsed_page(issues_page_url)
			issue_table = security_issues_html.find("div", attrs={"class": "js-navigation-container js-active-navigation-container"})
			if issue_table is None:
				print("No issues found for label: {}".format(label))
				continue
			issue_rows = issue_table.findChildren(recursive=False)
			for issue in issue_rows:
				issue_id = issue.attrs["id"][6:]
				download_issue(repo_name, issue_id, True, gt_file, label)
				num_security_issues += 1
	print("Found {} security-related issues".format(num_security_issues))
	if num_security_issues == 0:
		subprocess.call(['rm', '-rf', '{}/{}'.format(group_name, repo_name)])
		return

	security_labels = set(security_labels)
	# Then download an equal number of issues issues which do not have security tags
	num_non_security = 0
	parsed_issues_html = get_parsed_page(base_url + '/' + repo_name + '/issues?page=1&q=')
	n_pages = get_num_pages(parsed_issues_html)
	for page in range(1, n_pages + 1):

		issues_page_url = base_url + '/' + repo_name + '/issues?page={}&q='.format(page)
		parsed_issues_html = get_parsed_page(issues_page_url)
		issue_table = parsed_issues_html.find("div", attrs={"class": "js-navigation-container js-active-navigation-container"})
		if issue_table is None:
			issues_page_url = base_url + '/' + repo_name + '/pulls?page={}&q='.format(page)
			parsed_issues_html = get_parsed_page(issues_page_url)
			issue_table = parsed_issues_html.find("div", attrs={"class": "js-navigation-container js-active-navigation-container"})

		issue_rows = issue_table.findChildren(recursive=False)

		for issue in issue_rows:

			issue_labels_element = issue.find("span", attrs={"class": "labels"})
			if issue_labels_element is not None:
				has_security_label = False
				for label in issue_labels_element.findChildren():
					label = label.text.strip()
					if label in security_labels:
						has_security_label = True
						break
				if has_security_label:
					continue

			issue_id = issue.attrs["id"][6:]
			if num_non_security < num_security_issues:
				download_issue(repo_name, issue_id, False, gt_file)
				num_non_security += 1
			else:
				print("Finished downloading {} security issues and {} non-security issues for repo: {}".format(num_security_issues, num_non_security, repo_name))
				gt_file.close()
				return
	print("Finished downloading {} security issues and {} non-security issues for repo: {}".format(num_security_issues, num_non_security, repo_name))
	gt_file.close()

n_pages = get_num_pages(get_parsed_page(base_url))
for page in range(1, n_pages + 1):
	page_url = base_url + '?page=' + str(page)
	parsed_page_html = get_parsed_page(page_url)

	repos = [element.text.strip() for element in parsed_page_html.find_all("a", attrs={"itemprop":"name codeRepository"})]
	for repo in repos:
		print("Checking repo [{}] for security labels".format(repo))
		labels = get_repo_security_labels(base_url + '/' + repo)
		if len(labels) > 0:
			print("[{}] has labels: {}".format(repo, str(labels)))
			download_issues(repo, labels)
			print("Total issues downloaded: {}\n".format(global_num_issues))
		else:
			print("[{}] has no security labels\n".format(repo))

print("Final issue download count for ({}): {}".format(group_name, global_num_issues))
