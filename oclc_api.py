import os
from urllib.parse import quote
from dotenv import load_dotenv
import requests

load_dotenv()
client_id = os.environ["CLIENT_ID"]
client_secret = os.environ["CLIENT_SECRET"]


## Worldcat Search API v2.0
v2_auth_url = "https://oauth.oclc.org/token?grant_type=client_credentials&scope=wcapi"
v2_headers = {"Accept": "application/json"}

# Make the POST request
auth = requests.post(v2_auth_url, headers=v2_headers, auth=(client_id, client_secret))
# This returns a 403: "Invalid scope(s): wcapi (WorldCat Search API) [Service on WSKey does not permit all actions being requested]"
print(auth.text)


## Worldcat Search API v1.0
v1_search_url = "http://www.worldcat.org/webservices/"
v1_sru_endpoint = "catalog/search/sru?q="

wskey = f"&wskey={client_id}"
v1_headers = {"accept": "application/json"}

sru_query = 'srw.ti="FENG"'
sru_query_url = f"{v1_search_url}{v1_sru_endpoint}{quote(sru_query)}&format=rss&wskey={client_id}"

# Make the GET request
sru_req = requests.get(sru_query_url, headers=v1_headers)
print(sru_req.text)
# This returns an RSS xml with a diagnostic containing
#  <details>java.lang.NullPointerException</details>
#  <message>General System Error</message>