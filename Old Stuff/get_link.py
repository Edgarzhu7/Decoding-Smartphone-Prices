from bs4 import BeautifulSoup
import requests

# List of specific models to scrape
target_models = [
    "Asus ZenFone Live",
    "Intex Aqua Crystal",
    "Alcatel Pixi 4 (5)",
    "Reach Allure",
    "Karbonn Quattro L52",
    # Add more models here
]

# Function to get links to all pages of a brand
def get_pages(link, link_list):
    html = requests.get(link).content
    soup_page = BeautifulSoup(html, "lxml")
    pages = soup_page.select("div .nav-pages a")
    for page in pages:
        link_list.append("https://www.gsmarena.com/" + page['href'])

# Main function to get brand links
url = "https://www.gsmarena.com/makers.php3"
html = requests.get(url).content
soup = BeautifulSoup(html, 'lxml')
td_links = soup.select('tr td a')
brand_links = []
for anchor in td_links:
    link = "https://www.gsmarena.com/" + anchor['href']
    brand_links.append(link)
    get_pages(link, brand_links)
    
# Save brand links to a file (optional)
with open("brand_links.txt", "w") as f:
    for link in brand_links:
        f.write(f'{link}\n')

# Load brand links (you can control the number of pages to scrape by modifying the loop)
selected_brand_links = []
with open("brand_links.txt", "r") as f:
    for i in range(1):  # Change here to select how many pages to surf
        selected_brand_links.append(f.readline().strip())

# Function to check if the phone is in the target list
def is_target_model(name):
    return any(model.lower() in name.lower() for model in target_models)

# Scrape phone links for specific models only
phone_links = []
for link in selected_brand_links:
    html = requests.get(link).content
    soup = BeautifulSoup(html, 'lxml')
    phone_tags = soup.select('div.makers ul li a')
    for phone in phone_tags:
        phone_name = phone.text.strip()
        if is_target_model(phone_name):
            phone_link = "https://www.gsmarena.com/" + phone['href']
            phone_links.append(phone_link)
            print(f"Added link for {phone_name}")

# Save the specific phone links to a file (optional)
with open("phone_links.txt", "w") as f:
    for link in phone_links:
        f.write(f'{link}\n')
