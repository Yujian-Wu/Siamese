
# coding: utf-8

# In[1]:

import re
from bs4 import BeautifulSoup
import selenium
from selenium import webdriver
import time
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen


# In[3]:

driver = webdriver.Chrome()
driver.get("https://genuine-people.com/collections/tops")

# to get the full html
for i in range(10):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
text_html = BeautifulSoup(driver.page_source,"lxml")

all_tag = text_html.find_all('img')

num = 0

for tag in all_tag:
    new = str(tag)
    pattern = re.compile(r'<img.*?src="(.*?)\".*?>')
    match = pattern.match(new)
    if match:
        num += 1
        print (match.group(1))

# the number of pictures
print(num)


# In[ ]:



