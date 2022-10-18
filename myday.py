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



def main_page():
    st.markdown("# MyDay Admin 🎈")
    st.sidebar.markdown("# MyDay Admin 🎈")
    st.sidebar.write('High level application user statistics for admin purposes.')

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
    st.sidebar.markdown("# Meal Input 🍔")
    st.sidebar.write('Provide & grade meal information to be later used for ML Model training.')
    st.markdown("# Meal Input 🍔")

    col1, col2 = st.columns([2, 4])
    col1.markdown('<img src="https://github.com/mariusndini/myday-health-streamlit/blob/main/mastercalss-QR.png?raw=true" alt="drawing" width="200"/>', unsafe_allow_html=True)
    col2.markdown("##### Input data points below for a specific meal and grade it with respect to your diet goals. Thereafter a ML/AI model will be trained on the crowd sourced data.")

    import qrcode
    # Data to be encoded
    data = 'QR Code using make() function'
    
    # Encoding data using make() function
    img = qrcode.make(data)
    img.save('MyQRCode2.png')
    st.write(img.image.pil)
    st.markdown('<img src="MyQRCode2.png" alt="drawing" width="200"/>', unsafe_allow_html=True)

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
        save_SQL = f"""insert into sh_marius.pipeline_train.GRADED_DIET values(
                        '{modelName}',{cals},{carbs},{fat},{protein},{chol},{salt},{sugar},{grade},'{note}');"""
        st.write( f"""Rows insert: {run_query(save_SQL)[0][0]}""")

    st.markdown("""---""")
    if st.button('Show Data Points'):
        datapoints = run_query(f""" select * from sh_marius.pipeline_train.GRADED_DIET where DIET_NAME='{modelName}'; """)

        datapoints_df = pd.DataFrame(
                {
                    "Name": [i[0] for i in datapoints],
                    "Cals": [i[1] for i in datapoints],
                    "Carbs": [i[2] for i in datapoints],
                    "Fat": [i[3] for i in datapoints],
                    "Protein": [i[4] for i in datapoints],
                    "Chol": [i[5] for i in datapoints],
                    "Salt": [i[6] for i in datapoints],
                    "Sugar": [i[7] for i in datapoints],
                    "Grade": [i[8] for i in datapoints],
                    "Note": [i[9] for i in datapoints]
                }
        )

        df = st.dataframe(datapoints_df)

# ----------------------------------------------------------------------------------
# DONE INPUT PAGE ------------------------------------------------------------------
# ----------------------------------------------------------------------------------





def train():
    st.sidebar.markdown("# Train Model 🤖")
    st.sidebar.write('Model is trained from this page provided some high level inputs.')
    st.markdown("# Train Model 🤖")
    
    parems = st.experimental_get_query_params() or {"model":["DefaultModelName"]}
    modelName = parems['model'][0]

    c1, c2, c3 = st.columns(3)

    with c1:
        model_name = st.text_input('Model Name', modelName)
    with c2:
        iter = st.number_input('Training Iterations', min_value=1, step = 1)
    with c3:
        layers = st.text_input( "Enter Layers (array)", placeholder="e.g: 10,5,15")

    if st.button('Train Model'):
        grade_SQL = f""" call SH_MARIUS.PIPELINE_TRAIN.TRAIN_MODEL('{model_name}', {iter}, [{layers}]); """
        st.write( f"""Your Model: {run_query(grade_SQL)[0][0] }""")
    
    st.markdown("""---""")
    
    st.write(f'''Your model is being trained via the Python Stored Procedure below. 
                 This model is fully trained on Snowflake Compute Warehouse. 
                 The model is then deployed to a UDF to be used as needed.''')

    code = '''
CREATE OR REPLACE PROCEDURE TRAIN_MODEL(MODEL_NAME STRING, ITER INTEGER, LAYERS ARRAY)
    RETURNS STRING
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.8'
    PACKAGES = ('snowflake-snowpark-python', 'scikit-learn') 
    HANDLER = 'main'
    AS
    $$

    # Import Necessary Libraries Needed to Train Model
    import snowflake.snowpark as snowpark
    from snowflake.snowpark import Session
    from snowflake.snowpark.functions import udf
    from snowflake.snowpark.functions import col
    from snowflake.snowpark import functions as F
    from snowflake.snowpark.types import *
    from sklearn.neural_network import MLPClassifier

    def main(session: snowpark.Session, MODEL_NAME):
        # Model Characteristics
        mlpc = MLPClassifier(hidden_layer_sizes = (LAYERS), max_iter= ITER);
        
        # X values - From Snowflake Table
        x = session.sql(f"""SELECT CAL_DIFF, CARBS, FAT, PROTEIN, CHOL, SALT, SUGAR 
                            FROM SH_MARIUS.pipeline_train.GRADED_DIET
                            WHERE DIET_NAME = '{MODEL_NAME}';""").collect()

        # Y Values - From Snowflake Table
        y = session.sql(F"""SELECT GRADE 
                            FROM SH_MARIUS.pipeline_train.GRADED_DIET
                            WHERE DIET_NAME = '{MODEL_NAME}';""").collect()

        # Model Training - Fit the model to your data in Snowflake
        mlpc.fit(x, y) 

        # Define, Create Model & Create UDF With Trained Model
        @udf(name= f"""PREDICT_{MODEL_NAME}""", 
            return_type=IntegerType(), 
            packages=["scikit-learn"],
            is_permanent=True, replace=True, 
            stage_location="@MyStage",
            input_types=[FloatType(), FloatType(),FloatType(), FloatType(), FloatType(), FloatType(), FloatType()])
        
        def PREDICT_DIET(calAte, carbs, fat, protein, chol, salt, sugar):
            return mlpc.predict( [[calAte, carbs, fat, protein, chol, salt, sugar]] )[0]
    $$
        
        '''
    st.code(code, language='python')


# ----------------------------------------------------------------------------------
# DONE TRAIN PAGE ------------------------------------------------------------------
# ----------------------------------------------------------------------------------




def test():
    st.sidebar.markdown("# Test Models 🧠")
    st.sidebar.write('Test the trained models in Snowflake via this page.')
    st.markdown("# Test Models 🧠")

    save_SQL = f""" select distinct * from sh_marius.pipeline_train.model_list; """
    trained_models = run_query(save_SQL)

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
        grade_SQL = f"""select sh_marius.PIPELINE_TRAIN.PREDICT_{option.upper()}({cals},{carbs},{fat},{protein},{chol},{salt},{sugar});"""
        st.markdown( f"""# Your Meal Grade: {run_query(grade_SQL)[0][0] }""")
        st.write("Remember - A grade of 5 follows the Diet's rules and 1 does not.")
        st.markdown("""---""")
        st.write("The SQL to run the model is below - ")
        st.markdown(f"""<b style='color:#FFBF00'>{grade_SQL}</b>""", unsafe_allow_html=True)




# ----------------------------------------------------------------------------------
# DONE INPUT PAGE ------------------------------------------------------------------
# ----------------------------------------------------------------------------------



page_names_to_funcs = {
    "Admin Page": main_page,
    "Meal Input": input,
    "Train Model": train,
    "Test Models": test,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()






