from selenium import webdriver

webpage = r'https://www.tripadvisor.com/Attractions'
searchterm = 'austin'

driver = webdriver.Chrome()
driver.get(webpage)

sbox = driver.find_element_by_name('q')
sbox.send_keys(searchterm)

submit = driver.find_element_by_class_name('searchAttractionsBtn')
submit.click()

driver.quit()