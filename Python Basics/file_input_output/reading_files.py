import os
print(os.getcwd()) # Print the current working directory

#open file in the read mode
with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/test.txt', 'r') as file:
    content = file.read() # Read the entire content of the file
    print(content) # print the content of the file

#reading line by line
with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/example.txt', 'r') as file:
    for line in file:
        print(line.strip()) # print each line without extra newlines, strip removes any extra newline characters


#reading csv file
import csv
with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/Amazon_Reviews.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row) # print each row of the csv file


#writing to a file
#writing to a text file
with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/test.txt', 'w') as file:
    file.write('Hello, World! \n') # write to the file
    file.write('This is a new line.') # write a new line to the file
  

#writing to a file with append mode
with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/test.txt', 'a') as file:
    file.write('This is an appended line.') # append a new line to the file
    file.write('\n') # add a new line
    file.write('This is another appended line.') # append another new line to the file



#writing to a csv file
import csv
data = [['Name', 'Age', ],[ 'samir', 29], ['john',25]]
with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/Amazon_Reviews.csv', 'w', newline='') as file: #newline is used to prevent extra space between rows
    writer = csv.writer(file)
    writer.writerows(data) # write the data to the csv file
    print('Data written to CSV file successfully.')
    

    


