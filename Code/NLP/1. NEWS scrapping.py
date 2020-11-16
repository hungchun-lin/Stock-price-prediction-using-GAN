https://github.com/ASC521/googlefinance
https://towardsdatascience.com/scraping-the-aapl-stock-prices-using-python-431263c4b00c

"""
Author:TH
Date:17/05/2016
Download one artile using url.
Naming rules: http://stackoverflow.com/questions/2029358/should-i-write-table-and-column-names-always-lower-case
Table design: Title, Date, Time, TickersAbout, TickersIncludes, Name, NameLink, Bio, Summary, ImageDummy, BodyContent, Disclosure, Position
Time is under UTC
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
#from login import loginSA
import sys
import codecs
from time import sleep
from multiprocessing.pool import ThreadPool

"""
Author:TH
Date:16/05/2016
login to Seeking Alpha
When you try to get or post, please use a headers variable. otherwise you will be considered as a robot and receive a 403
A tutorial for BitBucket: https://kazuar.github.io/scraping-tutorial/
A really good YouTube tutorial: https://www.youtube.com/watch?v=eRSJSKG4mDA
"""



def collectArticle(url):
	# Set std out encoding
	#sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
	#print (sys.stdout)
	#sessionCode = loginSA()[0]
	#print(sessionCode)
	#session = loginSA()[1]

	userHeader = {"Referer": "http://seekingalpha.com/",
			"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"}

	r = requests.get(url, headers = userHeader)

	soup = BeautifulSoup(r.content, 'html.parser')
	sleep(2)
	###print(soup)
	"""
	file = open("out.txt","w")
	file.write(soup.prettify())
	file.close()
	"""
	"""
	poolSession = ThreadPool(processes=1)
	async_result_session_get = poolSession.apply_async(session.get, (url, userHeader))
	r = async_result_session_get.get()
	soup = BeautifulSoup(r.content, 'html.parser')
	"""

	pro = soup.find_all("div",{"class":"checkout-header-text"})
	if len(pro)!=0:
		print("Ignore a pro article.")
		return "pro: "+url
	try:
		title = soup.find_all("h1", {"itemprop":"headline"})[0].text
	except:
		print("Could not get title: ",url)
		return "Could not get title: "+url
	###print("title: ", title)
	dateTime = ''
	date = ''
	time = ''
	try:
		dateTime = soup.find_all("meta", {"property":"article:published_time"})[0]
		#time1 = dateTime.get("content")
		#time2 = dateTime.text
		date = dateTime.get("content").split('T')[0]
		time = dateTime.get("content").split('T')[1].split('Z')[0]
	except Exception as e:
		print("Could not get time: ", e)
	###print("Date time is: {0} and {1}".format(date, time))
	# This part, we could not collect the num of comments. we don't have this field when we download the webpage.
	# instead, we have a field with id="a-comments-wrapper"
	#numComments = soup.find_all("span", {id:"a-comments"})
	#print("Num of comments: ",numComments)

	#instead, I find another place for num of comments

	tickersAbout = []
	companiesAbout = soup.find_all("a", {"sasource":"article_primary_about"})
	for companyAbout in companiesAbout:
		if "(" in companyAbout.text:
			tickersAbout.append(companyAbout.text.split("(")[1].split(")")[0])
		else:
			tickersAbout.append(companyAbout.text)
	#print("Tickers About are: {0}".format(', '.join(tickersAbout)))
	tickersAboutStr = ', '.join(tickersAbout)

	tickersIncludes = []
	companiesIncludes = soup.find_all("a", {"sasource":"article_about"})
	for companyIncludes in companiesIncludes:
	    tickersIncludes.append(companyIncludes.text)
	###print("Tickers Includes are: {0}".format(', '.join(tickersIncludes)))
	tickersIncludesStr = ', '.join(tickersIncludes)

	author = soup.find_all("a",{"class":"name-link", "sasource":"auth_header_name"})
	authorUrl = author[0].get("href")
	authorName = author[0].contents[0].text
	###print("Name is: {0}, {1}".format(authorUrl, authorName))

	bio = soup.find_all("div", {"class":"bio hidden-print"})[0].text
	###print("Bio is: {0} ".format(bio))

	summary = []
	try:
		summaryByParagraphes = soup.find_all("div", {"class":"a-sum", "itemprop":"description"})[0].find_all("p")
		#print(summaryByParagraphes)
		for p in summaryByParagraphes:
		    summary.append(p.text);
		###print("Summary: ",' '.join(summary))
	except Exception as e:
		print(', No Summary ', url)
	summaryStr = ' '.join(summary)

	image = soup.find_all("span", {"class":"image-overlay"})
	if len(image) > 0:
	    imageDummy = 1
	else:
	    imageDummy = 0
	###print("ImageDummy: ",imageDummy)

	bodyAll = soup.find_all("div", {"id":"a-body"})[0]
	body = bodyAll.find_all("p")
	bodyContent = ''
	for p in body:
	    bodyContent += (p.text+' ')
	bodyContent = bodyContent.split("Disclosure")[0]
	bodyAll = bodyAll.text
	#print(bodyAll)


	try:
		disclosure = soup.find_all("p", {"id":"a-disclosure"})[0].find_all("span")[0].text
		###print("Disclosure: ", disclosure)
	except:
		try:
			disclosure = bodyAll.split("Disclosure:")[1]
		except:
			print(', No Disclosure ', url )
			disclosure = ''
	#print(disclosure)
	# New way of collecting disclosure.
	try:
		articleNumber = int(url.split('article/')[1].split("-")[0])
	except:
		print("No article number.")
	#print(articleNumber)
	return {"title": title,
			"date": date,
			"time": time,
			"tickersAbout": tickersAboutStr,
			"tickersIncludes": tickersIncludesStr,
			"name": authorName,
			"nameLink": authorUrl,
			"bio": bio,
			"summary": summaryStr,
			"bodyContent": bodyContent,
			"imageDummy": imageDummy,
			"disclosure": disclosure,
			"bodyAll": bodyAll,
			"articleNumber": articleNumber,
			'articleUrl2': url}

test = "http://seekingalpha.com/"
collectArticle(test)
