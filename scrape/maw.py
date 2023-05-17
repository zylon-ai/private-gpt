import requests
from bs4 import BeautifulSoup
import json
import os
import time
from urllib.parse import urljoin, urlparse

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_website_links(url):
    urls = set()
    domain_name = urlparse(url).netloc
    soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).content, "html.parser")

    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            continue
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        if not is_valid(href):
            continue
        if href in internal_urls:
            continue
        if domain_name not in href:
            if href not in external_urls:
                external_urls.add(href)
            continue
        urls.add(href)
        internal_urls.add(href)
    return urls

def crawl_site(url):
    global total_urls_visited
    links = get_all_website_links(url)
    data = {
        'url': url,
        'date': '',  # Needs the correct CSS selector
        'text': '',  # Needs the correct CSS selector
        'authors': [],  # Needs the correct CSS selector
        'external_links': list(external_urls),
        'citations': [],  # Needs the correct CSS selector
        'citation_links': []  # Needs the correct CSS selector
    }
    soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).content, "html.parser")

    # Assuming date is within a <time> tag
    date = soup.find('time')
    if date:
        data['date'] = date.text.strip()

    # Assuming text is within <p> tags
    text = soup.find_all('p')
    if text:
        data['text'] = ' '.join([t.text for t in text])

    # Assuming authors are in a <span class="author"> tag
    authors = soup.find_all('span', class_='author')
    if authors:
        data['authors'] = [author.text for author in authors]

    # Citations and citation links will depend on the website structure.

    # Save data as JSON
    if not os.path.exists('./scrap'):
        os.makedirs('./scrap')
    timestamp = int(time.time())
    with open(f'./scrap/data_{timestamp}.json', 'w') as f:
        json.dump(data, f)

    total_urls_visited += 1
    print(f"[+] Crawled {url}, Total Links Visited: {total_urls_visited}")

    for link in links:
        if total_urls_visited > max_urls:
            break
        crawl_site(link)

max_urls = 5000  # Define the maximum number of URLs to visit, change as needed.
total_urls_visited = 0
internal_urls = set()
external_urls = set()

url = input("Enter the URL to crawl: ")
crawl_site(url)
