import streamlit as st
import pandas as pd
import models.funding_model as model
from PIL import Image

@st.cache_resource
def get_functions():
    how_much_more, funding_optimizer,name_list = model.get_demo_functions()
    return how_much_more, funding_optimizer, name_list


@st.cache_data
def load_data():
    return pd.read_csv("out.csv")


def change_grad_rate(x):
    if x.find("GE") != -1:
        x = x.replace("GE", "")
    if x.find("LE") != -1:
        x = x.replace("LE", "")
    if x.find("GT") != -1:
        x = x.replace("GT", "")
    if x.find("LT") != -1:
        x = x.replace("LT", "")
    if x.find("-") != -1:
        x = x.split("-")[1]
    return f'{x}'
    
cols = [
    'School name',
    'District Name',
    'CATEGORY',
    'COHORT',
    'RATE',
    'Student enrollment',
    'free_and_reduced_lunch_students_2021_22',
    'free_lunch_eligible_2021_22',
    'reduced_price_lunch_eligible_students_2021_22',
    'full_time_equivalent_fte_teachers_2021_22',
    'pupil_teacher_ratio_2021_22',
    'Per-pupil expenditure - total personnel salaries'
]
new_cols = {
    'School name':'School Name',
    'District Name': 'District Name',
    'COHORT': 'Cohort',
    'RATE': 'Graduation Rate',
    'Student enrollment': 'Student Enrollment',
    'free_lunch_eligible_2021_22': 'Free Lunch Eligible Students',
    'full_time_equivalent_fte_teachers_2021_22': 'Full Time Teachers',
    'reduced_price_lunch_eligible_students_2021_22': 'Reduced Priced Lunch Students',
    'pupil_teacher_ratio_2021_22': 'Pupil to Teacher Ratio',
    'Per-pupil expenditure - total personnel salaries': 'Per-Pupil Expenditure',
    'free_and_reduced_lunch_students_2021_22': 'Free and Reduced Lunch Students'
}
df = load_data()
df = df[cols].loc[df['CATEGORY'] == 'ALL']
df = df.drop(columns='CATEGORY')
df = df.rename(columns=new_cols)
df = df.drop_duplicates()
df['Graduation Rate'] = df['Graduation Rate'].apply(lambda x: change_grad_rate(x))


st.title("Title I Funding")
st.subheader("Lousiana - Public High Schools")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Funding Optimizer", "Graduation Booster", "Graphics", "About"])

tab1.header("Overview")

tab1.subheader("Filters")
schools = df['School Name'].drop_duplicates()
disticts = df['District Name'].drop_duplicates()

school_options = tab1.multiselect(
    'Schools',
    df['School Name'].loc[(df['District Name'].isin(disticts))].drop_duplicates()
)

if len(school_options) > 0:
    schools = school_options

distict_options = tab1.multiselect(
    'Districts',
    df['District Name'].loc[(df['School Name'].isin(schools))].drop_duplicates()
)

if len(distict_options) > 0:
    disticts = distict_options


df = df.loc[(df['District Name'].isin(disticts)) & (df['School Name'].isin(schools))]

tab1.subheader("School General Information")
tab1.dataframe(df)

tab1.subheader("Funding allocation")
tab1.bar_chart(df[['School Name', 'Per-Pupil Expenditure']], x ='School Name', y='Per-Pupil Expenditure')

how_much_more, funding_optimizer,name_list = get_functions()

@st.cache_data
def funding_opt(input = 5000000):
    return funding_optimizer(input)

tab2.header("Funding Optimizer")
tab2.caption("Enter the total amount of funding that you wish to allocate to Title I schools in Lousiana. This will output the recommended set of funding per school that optimizes graduation rate under our model.")
user_input = tab2.number_input("Enter total funding:")

button_clicked = tab2.button("Submit")
if button_clicked:
    funding_df = funding_opt(int(user_input))
    tab2.subheader("Funding Distribution")
    tab2.write(funding_df)

tab3.header("Graduation Booster")
tab3.caption("Choose a specific school, a desired graduation rate to achieve for that school, and a metric. This will output what changes would be necessary for that school's metric to achieve the desired graduation rate under our model. This can help with informing how resources should be distributed.")

input_school = tab3.selectbox(
    'Schools',
    name_list
)
input_number = tab3.slider("Select a desired graduation rate:", min_value=0.0, max_value=1.0, value=0.9)
input_column = tab3.selectbox(
    'Metric',
    ["Per Pupil Funding","Pupil Teacher Ratio",'School Poverty Rate','Free Lunch Eligible Students']
)

input_dict = {"Per Pupil Funding": "per_pupil_funding", "Pupil Teacher Ratio": "pupil_teacher_ratio", "School Poverty Rate" : "School poverty rate", "Free Lunch Eligible Students": "free_lunch_eligible"}

button_clicked_two = tab3.button("Submit", key = "two")
if button_clicked_two:
    number = how_much_more(input_school,input_number,input_dict[input_column])
    tab3.subheader(f"{input_column}: :arrow_up_small: {number}" if number > 0 else f"{input_column}: :arrow_down_small: {number}")

tab4.header("Graphics")
image1 = Image.open('eda_and_visualizations/visualizations/schools.png')
tab4.image(image1, caption='Population Distribution in Schools')
image2 = Image.open('eda_and_visualizations/ed_data_express_visualizations/pairplot.png')
tab4.image(image2, caption='Correlation between Graduation Rate and School Improvement Fund')

tab5.header("About")
tab5.subheader("Contributor")
tab5.markdown("**Brittany Leslie**")
tab5.markdown("**Keese Phillips**")
tab5.markdown("**Steven Li**")
tab5.markdown("**Yancey Yang​**")

tab5.subheader("Repository")
tab5.markdown('<a href="https://github.com/keesephillips/aipi510_project" target="_blank">Source Code</a>', unsafe_allow_html=True)

tab5.subheader("Data Source")
tab5.markdown('<a href="https://eddataexpress.ed.gov/" target="_blank">Department of Education Data Express</a>', unsafe_allow_html=True)
tab5.markdown('<a href="https://nces.ed.gov/ccd/elsi/" target="_blank">National Center for Education Statistics</a>', unsafe_allow_html=True)
tab5.markdown('<a href="https://www.census.gov/programs-surveys/saipe/data/api.html" target="_blank">United States Census Bureau</a>', unsafe_allow_html=True)
tab5.markdown('<a href="https://www.louisiana.gov/education/" target="_blank">Louisiana Department of Education</a>', unsafe_allow_html=True)
tab5.markdown('<a href="https://data.ed.gov/" target="_blank">Department of Education Data Profiles</a>', unsafe_allow_html=True)
tab5.markdown('<a href="https://oese.ed.gov/ppe/louisiana/​" target="_blank">Office of Elementary and Secondary Education</a>', unsafe_allow_html=True)