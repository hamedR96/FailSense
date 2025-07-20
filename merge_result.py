import os

# Path to the folder with your .txt files
folder_path = './classifier_results'  # <- Replace with your actual folder path
output_file = 'classifier_merged_output.txt'

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as infile:
                outfile.write(f'--- {filename} ---\n')  # Write the file name
                outfile.write(infile.read())            # Write the file content
                outfile.write('\n\n')                   # Add spacing between files

print(f"All files merged into {output_file}")
