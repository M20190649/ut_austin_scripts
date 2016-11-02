from splinter import Browser                
with Browser() as browser: 
  browser.visit(url)
  browser.fill('textbox', 'your text here')
  browser.find_by_name('Submit').click()
  copied_text = browser.find_by_id('results')[0].text
