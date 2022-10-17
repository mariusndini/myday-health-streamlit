import streamlit as st
import snowflake.connector
import pandas as pd
from PIL import Image


conn = snowflake.connector.connect( user= st.secrets["user"],
                                    password= st.secrets["pw"],
                                    account= st.secrets["account"],
                                    role = st.secrets["role"],
                                    warehouse = 'streamlit',
                                    session_parameters={
                                        'QUERY_TAG': 'Streamlit',
                                    })

# function to run queries on Snowflake
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

conn_worksheets = snowflake.connector.connect( user= st.secrets["user_ws"],
                                password= st.secrets["pw_ws"],
                                account= st.secrets["acct_ws"],
                                role = st.secrets["role_ws"],
                                warehouse = 'streamlit',
                                session_parameters={
                                    'QUERY_TAG': 'Streamlit',
                                })
def run_ws(query):
    with conn_worksheets.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()




def main_page():
    st.markdown("# MyDay Admin 🎈")
    st.sidebar.markdown("# MyDay Admin 🎈")

    pop_count = run_query(f"""
        select count(distinct id) as pop
        ,avg(age)::integer as age
        ,count_if(gender='Male') as m
        ,count_if(Gender='Female') as f
        from SNOWHEALTH.HK.POPULATION
    """)


    metrics = st.columns(4)
    metrics[0].metric(label="📱 Total Users", value=pop_count[0][0])
    metrics[1].metric(label="👴 Avg User Age", value=pop_count[0][1])
    metrics[2].metric(label="👦 Male Users", value=pop_count[0][2])
    metrics[3].metric(label="👧 Female Users", value=pop_count[0][3])


    population = run_query(f"""
        select *, datediff(day, LOADTIME, CURRENT_DATE) as d
        from SNOWHEALTH.HK.POPULATION
        order by d asc
    """)

    pop_df = pd.DataFrame(
            {
                "ID": [i[0] for i in population],
                "Age": [i[1] for i in population],
                "Bloodtype": [i[2] for i in population],
                "Gender": [i[3] for i in population],
                "Last Load": [i[4] for i in population],
                "Days Since Load": [i[5] for i in population]
            }
    )

    df = st.dataframe(pop_df)




# ----------------------------------------------------------------------------------
# DONE myday admin PAGE ------------------------------------------------------------------
# ----------------------------------------------------------------------------------









def input():
    st.sidebar.markdown("# Diet Input 🍔")

    st.markdown("# Diet Input 🍔")
    st.markdown("##### Below input data points for a meal and grade it. Thereafter a ML/AI model will be trained on the crowd sourced data.")

    st.markdown("![Alt Text](https://github.com/mariusndini/myday-health-streamlit/blob/main/mastercalss-QR.png?raw=true)")


    parems = st.experimental_get_query_params() or {"model":["DefaultModelName"]}
    modelName = parems['model'][0]
    st.markdown(f"""### Model Name: {modelName}""")

    st.markdown("""---""")
    # cals input
    cals = st.slider('How many calories is the Meal?', 10, 1000, 25)
    st.markdown("""---""")
    carbs = st.slider('Carbs (g)', 0, 50, 5)
    st.markdown("""---""")
    fat = st.slider('fat (g)', 0, 50, 5)
    st.markdown("""---""")
    protein = st.slider('protein (g)', 0, 50, 5)
    st.markdown("""---""")
    chol = st.slider('chol (g)', 0, 50, 5)
    st.markdown("""---""")
    salt = st.slider('salt (g)', 0, 50, 5)
    st.markdown("""---""")
    sugar = st.slider('Sugar (g)', 0, 50, 5)
    st.markdown("""---""")

    st.write("Grade the Meal above from 1(worst) - 5 (healthiest)")
    grade = st.selectbox('Please Grade The Above Meal', ('1','2','3','4','5'))
    
    st.markdown("""---""")
    note = st.text_input('Diet Notes', '')


    if st.button('Save Graded Diet'):
        save_SQL = f"""insert into SNOWHEALTH_LOCAL.pipeline_train.GRADED_DIET values(
                        '{modelName}',{cals},{carbs},{fat},{protein},{chol},{salt},{sugar},{grade},'{note}');"""
        st.write( f"""Rows insert: {run_ws(save_SQL)[0][0]}""")





def test():
    st.sidebar.markdown("# Test Models 🧠")
    st.markdown("# Test Models 🧠")

    save_SQL = f""" select * from snowhealth_local.pipeline_train.model_list; """
    trained_models = run_ws(save_SQL)

    option = st.selectbox('Please Select a Trained Model Below', [i[0] for i in trained_models])

    cals = st.slider('How many calories did you eat?', 10, 1000, 25)
    st.markdown("""---""")
    carbs = st.slider('How many Carbs (g)?', 0, 50, 5)
    st.markdown("""---""")
    fat = st.slider('How much fat (g)?', 0, 50, 5)
    st.markdown("""---""")
    protein = st.slider('How much protein (g)?', 0, 50, 5)
    st.markdown("""---""")
    chol = st.slider('How much chol (g)?', 0, 50, 5)
    st.markdown("""---""")
    salt = st.slider('How much salt (g)?', 0, 50, 5)
    st.markdown("""---""")
    sugar = st.slider('How many Sugar (g)?', 0, 50, 5)
    st.markdown("""---""")

    if st.button('Grade My Diet'):
        grade_SQL = f"""select SNOWHEALTH_LOCAL.PIPELINE_TRAIN.PREDICT_{option.upper()}({cals},{carbs},{fat},{protein},{chol},{salt},{sugar});"""
        st.markdown( f"""# Your Diet Grade: {run_ws(grade_SQL)[0][0] }""")
        st.write("Remember - A grade of 5 follows the Diets rules and 1 does not.")
        st.markdown("""---""")
        st.write("The SQL to run the model is below - ")
        st.markdown(f"""<b style='color:#FFBF00'>{grade_SQL}</b>""", unsafe_allow_html=True)




# ----------------------------------------------------------------------------------
# DONE INPUT PAGE ------------------------------------------------------------------
# ----------------------------------------------------------------------------------



page_names_to_funcs = {
    "Admin Page": main_page,
    "Diet Input": input,
    "Test Models": test,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()






