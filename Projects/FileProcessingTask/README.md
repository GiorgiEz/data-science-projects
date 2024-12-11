# Student Grades Management Project

## Overview
This project manages student grades across multiple subjects. It includes features for generating random grades, storing data in files, processing grades, calculating averages, and identifying the top-performing student.

## Features
1. **Generate Random Data**: Creates random grades for five students across three subjects.
2. **File Management**:
   - `students.csv`: Stores student names and their grades.
   - `subjects.txt`: Lists the names of subjects.
   - `top_student.txt`: Contains details of the top-performing student.
3. **Process Data**: Reads and organizes data from CSV and text files.
4. **Calculate Averages**: Computes the average grade for each student.
5. **Identify Top Student**: Finds the student with the highest average grade and writes their details to a file.

## Prerequisites
1. Python 3
2. Required libraries:
   - `csv`
   - `random`
   - `os`

## Directory Structure
```
project_root/
├── FileProcessing.py
├── students.csv
├── subjects.txt
├── top_student.txt
└── README.md
```

## How to Use

### 1. Run the Script
Execute the script by running:
```bash
python FileProcessing.py
```

### 2. Output Files
After running the script, the following files will be created or updated:
- **`students.csv`**: Contains names of students and their randomly generated grades.
- **`subjects.txt`**: Lists the subjects (Math, Science, History).
- **`top_student.txt`**: Details of the student with the highest average grade.

### 3. View Results
The console will display any errors or key operations. The top student's information will also be saved in `top_student.txt`.

## Code Breakdown

### Functions
- **`generate_students_data()`**
  Generates random grades for five students in three subjects.

- **`generate_files()`**
  Creates and populates `students.csv` with student names and grades, and `subjects.txt` with subject names.

- **`process_data()`**
  Reads and organizes data from `students.csv` and `subjects.txt`.

- **`calculate_average()`**
  Computes the average grade for each student.

- **`identify_top_student()`**
  Finds the student with the highest average grade.

- **`write_top_student_to_file()`**
  Writes the name, average grade, and individual grades of the top student to `top_student.txt`.

### Entry Point
The `if __name__ == '__main__':` block executes the following steps sequentially:
1. Generates random student data.
2. Creates required files.
3. Processes data from files.
4. Calculates average grades.
5. Identifies and records the top student.


