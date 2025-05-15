# Write a program that:
# Reads from a text file and prints each line.
# Writes some text to a new file.
# Appends additional text to that file.


#task 1 --> read
with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/example.txt', 'r') as file:
    content = file.read()
    print(content)

#task 1 --> write

with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/example.txt', 'w') as file:
    file.write("This is a new line of text.\n")
    file.write("This is another line of text.\n")

#task 1 -->  append
with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/example.txt', 'a') as file:
    file.write("Appendint this text, \n")
    file.write('Appending other line in this text file,\n')

#verifying the read file
with open('/Users/samirsitaula/Documents/Selfpaced_Practice/Python Basics/file_input_output/example.txt', 'r') as file:
    content = file.read()
    print(content)

