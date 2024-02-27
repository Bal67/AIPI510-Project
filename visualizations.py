import sys
# needed to import from sources
sys.path.append('..')
import pandas as pd
import matplotlib.pyplot as plt
from data.sources import Sources

"""
Script to create visualizations on fundamental fields as part of data preperation and determine
the underlying trends within the data and possible areas of note.

All of the visualization charts can be found in the visualizations folder

"""

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')
        
if __name__=="__main__": 
    data = Sources("localhost", "aipi510_project", "root", "password") 

    query = """
    SELECT * FROM AIPI510_PROJECT.SSLE AS SSLE

    INNER JOIN AIPI510_PROJECT.SAIPE AS SAIPE
    ON SAIPE.`LEA ID` = SSLE.`LEA ID`
    
    INNER JOIN aipi510_project.graduation_rate_category AS GRAD_CAT
    ON GRAD_CAT.`NCESSCH` = SSLE.`NCES ID` AND GRAD_CAT.`CATEGORY` = 'ALL' AND GRAD_CAT.`year` = SAIPE.`year`
    
    INNER JOIN AIPI510_PROJECT.NCES_BY_YEAR AS NCES
    ON NCES.`school_id_nces_assigned` = SSLE.`NCES ID` AND NCES.`year` = SAIPE.`year`
    
    INNER JOIN AIPI510_PROJECT.title_i_max_funding AS funding
    on funding.`LEA ID` = SSLE.`LEA ID` AND funding.`year` = SAIPE.`year`
    
    INNER JOIN AIPI510_PROJECT.per_pupil_data as per_pupil_data
    on per_pupil_data.`site_name` = SSLE.`School name` AND per_pupil_data.`year` = SAIPE.`year`
    WHERE SSLE.`Title I eligibility`="Title I" AND SAIPE.`year` in ('2020') AND per_pupil_data.`reporting_category` = "Per Pupil Expenses"

    """
    df = pd.read_sql_query(query, data.engine)
    print(df)

    df['total_students_all_grades_excludes_ae'] = df ['total_students_all_grades_excludes_ae'].astype(int, errors='ignore')
    student_population = df[['school_name', 'total_students_all_grades_excludes_ae']]
    student_population = student_population.sort_values(by='total_students_all_grades_excludes_ae', ascending=False).drop_duplicates()
    student_population = student_population.iloc[:10]
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    plt.xticks(rotation='vertical')
    ax.set_title("Largest School Populations")
    ax.set_xlabel('School')
    ax.set_ylabel('Student Population')
    ax.bar(student_population['school_name'], student_population['total_students_all_grades_excludes_ae'])
    plt.savefig('visualizations/total_school_population.png')

    df['free_lunch_eligible'] = df['free_lunch_eligible'].astype(int, errors='ignore')
    free_lunch = df[['school_name', 'free_lunch_eligible']]
    free_lunch = free_lunch.sort_values(by='free_lunch_eligible', ascending=False).drop_duplicates()
    free_lunch = free_lunch.iloc[:10]
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    plt.xticks(rotation='vertical')
    ax.set_title("Free Lunch Eligible Students")
    ax.set_xlabel('School')
    ax.set_ylabel('Student Population')
    ax.bar(free_lunch['school_name'], free_lunch['free_lunch_eligible'])
    plt.savefig('visualizations/free_lunch_students.png')

    df['pupil_teacher_ratio'] = df['pupil_teacher_ratio'].astype(float, errors='ignore')
    student_to_teacher = df[['school_name', 'pupil_teacher_ratio']]
    student_to_teacher = student_to_teacher.sort_values(by='pupil_teacher_ratio', ascending=False).drop_duplicates()
    student_to_teacher_highest = student_to_teacher.iloc[:10]
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    plt.xticks(rotation='vertical')
    ax.set_title("Students to Teachers Ratio (Highest)")
    ax.set_xlabel('School')
    ax.set_ylabel('Students to Teachers Ratio')
    ax.bar(student_to_teacher_highest['school_name'], student_to_teacher_highest['pupil_teacher_ratio'])
    plt.savefig('visualizations/student_to_teachers_highest.png')

    student_to_teacher = student_to_teacher.sort_values(by='pupil_teacher_ratio', ascending=True)
    student_to_teacher = student_to_teacher.iloc[:10]
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    plt.xticks(rotation='vertical')
    ax.set_title("Students to Teachers Ratio (Lowest)")
    ax.set_xlabel('School')
    ax.set_ylabel('Students to Teachers Ratio')
    ax.bar(student_to_teacher['school_name'], student_to_teacher['pupil_teacher_ratio'])
    plt.savefig('visualizations/student_to_teachers_lowest.png')
    
    df['full_time_equivalent_fte_teachers'] = df['full_time_equivalent_fte_teachers'].astype(float, errors='ignore')
    teacher_full_time = df[['school_name', 'full_time_equivalent_fte_teachers']]
    teacher_full_time = teacher_full_time.sort_values(by='full_time_equivalent_fte_teachers', ascending=False).drop_duplicates()
    teacher_full_time = teacher_full_time.iloc[:10]
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    plt.xticks(rotation='vertical')
    ax.set_title("Full Time Teachers (Highest)")
    ax.set_xlabel('School')
    ax.set_ylabel('Full Time Teachers')
    ax.bar(teacher_full_time['school_name'], teacher_full_time['full_time_equivalent_fte_teachers'])
    plt.savefig('visualizations/teachers_highest.png')
    
    teacher_full_time = df[['school_name', 'full_time_equivalent_fte_teachers']]
    teacher_full_time = teacher_full_time.sort_values(by='full_time_equivalent_fte_teachers', ascending=True).drop_duplicates()
    teacher_full_time = teacher_full_time.iloc[:10]
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    plt.xticks(rotation='vertical')
    ax.set_title("Full Time Teachers (Lowest)")
    ax.set_xlabel('School')
    ax.set_ylabel('Full Time Teachers')
    ax.bar(teacher_full_time['school_name'], teacher_full_time['full_time_equivalent_fte_teachers'])
    plt.savefig('visualizations/teachers_lowest.png')

    schools_df = df[['school_name', 'full_time_equivalent_fte_teachers', 'free_lunch_eligible','total_students_all_grades_excludes_ae']]
    schools_df = schools_df.sort_values(by=['free_lunch_eligible', 'full_time_equivalent_fte_teachers', 'total_students_all_grades_excludes_ae'], ascending=False).drop_duplicates()
    schools_df = schools_df.iloc[:10]
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    plt.xticks(rotation='vertical')
    ax.set_title("Schools")
    ax.set_ylabel('Population')
    rects1 = ax.bar(teacher_full_time['school_name'], schools_df['total_students_all_grades_excludes_ae'], color='limegreen')
    rects2 = ax.bar(teacher_full_time['school_name'], schools_df['free_lunch_eligible'], color='forestgreen')
    rects3 = ax.bar(teacher_full_time['school_name'], schools_df['full_time_equivalent_fte_teachers'], color='darkgreen')
    ax.legend( (rects1[0], rects2[0], rects3[0]), ('Total Students', 'Free Lunch Students', 'Full Time Teachers'), loc='upper center' )
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    plt.savefig('visualizations/schools.png')
