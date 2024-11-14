import csv
import random
import os

# File paths
students_csv_path = "students.csv"
subjects_txt_path = "subjects.txt"
top_student_txt_path = "top_student.txt"

def generate_students_data():
    """
    Generates random grades for 5 students in 3 subjects.
    """
    return [
        ["Alice", random.randint(70, 100), random.randint(70, 100), random.randint(70, 100)],
        ["Bob", random.randint(70, 100), random.randint(70, 100), random.randint(70, 100)],
        ["Charlie", random.randint(70, 100), random.randint(70, 100), random.randint(70, 100)],
        ["David", random.randint(70, 100), random.randint(70, 100), random.randint(70, 100)],
        ["Eva", random.randint(70, 100), random.randint(70, 100), random.randint(70, 100)]
    ]

def generate_files():
    """
    Creates 'students.csv' with student names and grades, and 'subjects.txt' with subject names.
    """
    try:
        with open(students_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Math", "Science", "History"])  # header row
            writer.writerows(students_data)

        with open(subjects_txt_path, "w") as file:
            for subject in ["Math", "Science", "History"]:
                file.write(subject + "\n")
    except Exception as e:
        print(f"Error generating files: {e}")

def process_data():
    """
    Reads 'students.csv' and 'subjects.txt', assigning grades to each student.
    """
    student_grades = {}
    try:
        # Check if files exist
        if not os.path.exists(students_csv_path):
            raise FileNotFoundError(f"File {students_csv_path} not found.")
        if not os.path.exists(subjects_txt_path):
            raise FileNotFoundError(f"File {subjects_txt_path} not found.")

        # Read subjects from 'subjects.txt'
        with open(subjects_txt_path, "r") as file:
            subjects = [line.strip() for line in file.readlines()]

        # Read students data from 'students.csv'
        with open(students_csv_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Assign grades to subjects
                student_grades[row['Name']] = {}  # An empty dictionary for the student's grades
                for subject in subjects:
                    if not row[subject]:
                        raise ValueError(f"Student {row['Name']} has incorrect number of grades.")
                    student_grades[row['Name']][subject] = int(row[subject])

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ValueError as val_error:
        print(val_error)
    except Exception as e:
        print(f"An error occurred while processing data: {e}")
    return student_grades

def calculate_average():
    """
    Calculates the average grade for each student.
    """
    student_avg = {}
    for student, grades in student_grades.items():
        grade_sum = 0
        for subject, grade in grades.items():
            grade_sum += grade
        student_avg[student] = round(grade_sum / len(grades), 2)
    return student_avg

def identify_top_student():
    """
    Finds the student with the highest average grade.
    """
    result = {"Name": "", "Grade": 0}
    for student, avg_grade in student_avg.items():
        if avg_grade >= result["Grade"]:
            result["Name"] = student
            result["Grade"] = avg_grade
    return result

def write_top_student_to_file():
    """
    Writes the top student's name, average grade, and individual subject grades to 'top_student.txt'.
    """
    try:
        with open(top_student_txt_path, mode="w", newline="") as file:
            file.write(f'Top Student: {top_student["Name"]}' + "\n")
            file.write(f'Average Grade: {top_student["Grade"]}' + "\n")

            for subject, grade in student_grades[top_student["Name"]].items():
                file.write(f'{subject}: {grade}' + "\n")
    except Exception as e:
        print(f"Error writing top student data: {e}")


if __name__ == '__main__':
    """ Step 1: Generate Files """
    students_data = generate_students_data()
    generate_files()

    """ Step 2: Process Data """
    student_grades = process_data()

    """ Step 3: Calculate average """
    student_avg = calculate_average()

    """ Step 4: Identify Top Student """
    top_student = identify_top_student()
    write_top_student_to_file()
