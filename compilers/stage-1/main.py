from Lexer import *
import sys
import os
import glob

def process_file(file_path):
    print(f"\nProcessing file: {os.path.basename(file_path)}")
    print("-" * 40)
    
    try:
        lexer = Lexer(file_path)
        token = lexer.scan()
        while token.tag != Tag.EOF:
            print(str(token))
            token = lexer.scan()
        print("END")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def run_all_tests(script_dir):
    # Process all files in test_cases/bad
    print("===== PROCESSING BAD TEST CASES =====")
    bad_files_path = os.path.join(script_dir, "test_cases", "bad", "*.txt")
    bad_files = sorted(glob.glob(bad_files_path))
    
    for file_path in bad_files:
        process_file(file_path)
    
    # Process all files in test_cases/good
    print("\n===== PROCESSING GOOD TEST CASES =====")
    good_files_path = os.path.join(script_dir, "test_cases", "good", "*.txt")
    good_files = sorted(glob.glob(good_files_path))
    
    for file_path in good_files:
        process_file(file_path)

def list_available_files(script_dir):
    print("\nAvailable test files:")
    print("BAD TEST CASES:")
    bad_files_path = os.path.join(script_dir, "test_cases", "bad", "*.txt")
    bad_files = sorted(glob.glob(bad_files_path))
    for i, file_path in enumerate(bad_files):
        print(f"  {i+1}. bad/{os.path.basename(file_path)}")
    
    print("\nGOOD TEST CASES:")
    good_files_path = os.path.join(script_dir, "test_cases", "good", "*.txt")
    good_files = sorted(glob.glob(good_files_path))
    for i, file_path in enumerate(good_files, start=len(bad_files)+1):
        print(f"  {i}. good/{os.path.basename(file_path)}")
    
    return bad_files + good_files

def run_specific_file(files, file_number):
    if 1 <= file_number <= len(files):
        process_file(files[file_number-1])
    else:
        print("Invalid file number.")

if __name__ == '__main__':
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        print("\n=== LOGO DIALECT LEXER ===")
        print("1. Run all test cases")
        print("2. Run a specific test case")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            run_all_tests(script_dir)
        elif choice == '2':
            files = list_available_files(script_dir)
            file_choice = input("\nEnter the file number to process: ")
            try:
                file_number = int(file_choice)
                run_specific_file(files, file_number)
            except ValueError:
                print("Please enter a valid number.")
        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 3.")