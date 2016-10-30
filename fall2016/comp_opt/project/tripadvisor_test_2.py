from selenium import webdriver
import os

# chromedriver = '/home/paul/Downloads/chromedriver'
# os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome()
driver.implicitly_wait(10) # this lets webdriver wait 10 seconds for the website to load
driver.get("http://google.com")
print driver.title

barcodes = ["123451231", "6789021313", "231927813"]
for barcode in barcodes:
    text_box = driver.find_element_by_css_selector('#input') # input selector
    text_box.send_keys(barcode) # enter text in input

    driver.find_element_by_css_selector('#submit').click() # click the submit button

driver.quit()  