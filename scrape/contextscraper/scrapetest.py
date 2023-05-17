import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import ssl
import requests


import requests

def extract_data():
  """Extracts data from the homepage of https://www.theinformedslp.com/."""

  page = requests.get("https://www.theinformedslp.com/")
  soup = BeautifulSoup(page.content, "html.parser")

  texts = []
  authors = []
  titles = []
  pub_dates = []
  link_urls = []

  posts = soup.find_all(".post")

  for post in posts:
    title = post.find("h2").text.strip()
    author = post.find("span", {"class": "author"}).text.strip()
    date = post.find("small").text.strip()
    body = ""
    elements = post.find_all("div", "regular")

    # Extract and join all text paragraphs within the article together.
    for element in elements:
      p_elements = element.find_all("p")
      p_elements = list(filter(lambda x: len(x.strip()) > 0, p_elements))

      if len(body) == 0:
        body += str(next(iter(p_elements)))
      else:
        body += "<br><b> </b>" + next(iter(p_elements)).replace("\n ", "\n\n")

    body = body + "<p style='margin: 0px;'>...</p>"

    html = BeautifulSoup("<html><head></head><body>" + body + "</body></html>", "HTML")

    texts.append(text)
    authors.append(author)
    titles.append(title)
    pub_dates.append(date)
    link_urls.append((title, date, pub_date, SysAdmin.lookup_place))

  df = pd.DataFrame({
    "Title": titles,
    "Publication Date": pub_dates,
    "Author": authors,
    "Link URL": link_urls,
  })

  return df


# Call the function to extract data.
data = extract_data()

# Print the scraped data.
print(data)
