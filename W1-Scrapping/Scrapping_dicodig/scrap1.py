from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup   
import json
def scrapper(url):
    print("BEGIN TO SCRAP", url)
    try:
        # Configure WebDriver to use headless Firefox
        options = Options()
        options.add_argument('-headless')
        driver = webdriver.Firefox(options=options)
        
        
        # Get the URL given
        driver.get(url)
        # Selenium will wait for a maximum of 5 seconds for an element matching the given criteria to be found. 
        # If no element is found in that time, Selenium will raise an error.
        try:
            print('found')
            wait = WebDriverWait(driver, timeout=5)
            wait.until(EC.presence_of_element_located((By.ID, 'course-list')))
            
            # BeautifulSoup will parse the URL
            content = driver.page_source
            soup = BeautifulSoup(content, 'html.parser')
    
            # Prepare the variable for JSON data
            courses = []
            
            iter = 1
            # BeautifulSoup will find the CSS class that contain product container
            for course in soup.find_all('div', class_='col-md-6 mb-3'):
                print('course', iter)
                iter+=1
                
                # Get the text from the specified element and assign them to the variables
                course_name = course.find('h5', class_='course-card__name').text
                course_hour = course.find_all('span', {'class':'mr-2'})[0].text
                course_summary = course.select('div.course-card__summary p')[0].text
                course_total_module = course.find_all('div', class_= 'course-card__info-item')[0].find_all('span')[0].contents[0]
                course_level = course.find('span', attrs={'class': None}).text
                
                # Not all courses in the list has rating, so we should manage it. 
                # If it has rating, get the text. If none, set it to empty string.
                try:
                    course_rating = course.find_all('span', {'class':'mr-2'})[1].text
                except IndexError:
                    # Handle the case when no span elements with the specified class are found
                    course_rating = ''
    
                # Not all courses in the list has total students, so we should manage it. 
                # If it has total students, get the text. If none, set it to empty string.
                try:
                    course_total_students = course.find_all('span', {'class':'mr-3'})[1].get_text()
                except:
                    course_total_students = ''
                
                # Append the scraped data into courses variable for JSON data
                courses.append(
                    {
                        'Course Name': course_name,
                        'Learning Hour': course_hour,
                        'Rating': course_rating,
                        'Level': course_level,
                        'Summary': course_summary,
                        'Total Modules': course_total_module,
                        'Total Students': course_total_students
                    }
                )
            
                            # Close the WebDriver
            driver.quit()
    
            
            return courses
        except:
            raise LookupError("There is no element specified")

    
    except Exception as e:
        # eror invalid
        print("EROR", e)
        
    
    
    

if __name__ == '__main__':
    print('Mulai')
    url = "https://www.dicoding.com/academies/list"
    hasil = scrapper(url)
    
        # Save data to JSON file
    with open('dicoding_data.json', 'w') as json_file:
        json.dump(hasil, json_file, indent=4)