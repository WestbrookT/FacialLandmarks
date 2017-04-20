import requests as r, shutil
from bs4 import BeautifulSoup

query_string = 'https://www.google.com/search?q={}&tbm=isch'

phrases = ['empty+office', 'nature', 'empty+street', 'empty+house', 'empty+room', 'empty+basement', 'white+walls', 'forest']

img_urls = []

for item in phrases:
    page = r.get(query_string.format(item))


    soup = BeautifulSoup(page.text, 'lxml')
    for tag in soup.find_all('img'):
        img_urls.append(tag['src'])


for i, url in enumerate(img_urls):
    print(i, len(img_urls))

    response = r.get(url, stream=True)
    with open('nface/{}.jpg'.format(i), 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response