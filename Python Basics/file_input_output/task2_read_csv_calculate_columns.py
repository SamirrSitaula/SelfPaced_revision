import csv
with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/Amazon_Reviews.csv', 'r') as file:
    content = csv.reader(file)
    for row in content:
        print(row)
    for columns in content:
        print(columns)
        
   