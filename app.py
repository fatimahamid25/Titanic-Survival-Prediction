
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Titanic AI Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================= DARK TEAL THEME =================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Montserrat:wght@300;400;500;600&display=swap');

.stApp {
    background: radial-gradient(ellipse at top, #164545 0%, #0B2929 45%, #061B1B 100%);
    color: #D8E8E2;
    font-family: 'Montserrat', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2929, #041515);
    border-right: 1px solid #26766D;
}

[data-testid="stSidebar"] h1 {
    font-family: 'Cinzel', serif;
    color: #63D5BE !important;
    letter-spacing: 2px;
}

[data-testid="stSidebar"] p {
    color: #A7C9BE;
}

h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    color: #63D5BE !important;
}

p, label {
    color: #D8E8E2;
}

hr {
    border-color: #28665F;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #164545, #0B2929);
    border: 1px solid #26766D;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.25);
}

div[data-testid="stMetricLabel"] {
    color: #A7C9BE;
}

div[data-testid="stMetricValue"] {
    color: #63D5BE;
    font-family: 'Cinzel', serif;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #36B6A3, #197C70);
    color: #041B19;
    border: 1px solid #63D5BE;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #63D5BE;
    color: #041B19;
    border: 1px solid #63D5BE;
}

[data-testid="stForm"] {
    background: rgba(15, 51, 49, 0.85);
    border: 1px solid #26766D;
    border-radius: 12px;
    padding: 20px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #26766D;
    border-radius: 10px;
}

[data-baseweb="select"] > div {
    background-color: #164545;
    border-color: #26766D;
}

input, textarea {
    background-color: #164545 !important;
    color: #D8E8E2 !important;
}

.titanic-banner {
    background: linear-gradient(
        90deg,
        rgba(4,21,21,0.97),
        rgba(15,57,54,0.94),
        rgba(4,21,21,0.97)
    );
    border-top: 1px solid #36B6A3;
    border-bottom: 1px solid #36B6A3;
    padding: 38px 20px;
    text-align: center;
    border-radius: 12px;
    margin-bottom: 25px;
}

.titanic-banner h1 {
    font-family: 'Cinzel', serif;
    font-size: 34px;
    color: #63D5BE;
    letter-spacing: 4px;
}

.titanic-banner p {
    color: #A7C9BE;
    letter-spacing: 2px;
    font-size: 12px;
}

.small-caption {
    color: #82BDB0;
    letter-spacing: 2px;
    font-size: 11px;
}

</style>
""", unsafe_allow_html=True)


# ================= FILE PATHS =================

MODEL_PATH = "titanic_final_model.pkl"
DATA_PATH = "titanic_processed.csv"

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize"
]


# ================= LOAD MODEL =================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


# ================= LOAD DATA =================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    if "FamilySize" not in data.columns:
        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    return data


try:

    model = load_model()
    df = load_data()

    required_columns = FEATURES + ["Survived"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error("Some required columns are missing from your CSV.")
        st.write(missing_columns)
        st.stop()

except Exception as e:

    st.error("Could not load the Titanic project files.")
    st.write("Make sure the model and CSV are in the same folder as app.py.")
    st.code(str(e))
    st.stop()


# ================= HELPER FUNCTIONS =================

def banner(title, subtitle):

    st.markdown(f"""
    <div class="titanic-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def chart_layout(fig, title=None):

    fig.update_layout(
        title=title,
        paper_bgcolor="#0B2929",
        plot_bgcolor="#0B2929",
        font=dict(
            color="#D8E8E2",
            family="Montserrat"
        ),
        title_font=dict(
            color="#63D5BE",
            size=18
        ),
        xaxis=dict(
            gridcolor="#24504B"
        ),
        yaxis=dict(
            gridcolor="#24504B"
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )

    return fig


def predict_passenger(passenger):

    passenger_df = pd.DataFrame(
        [passenger],
        columns=FEATURES
    )

    prediction = model.predict(passenger_df)[0]
    probabilities = model.predict_proba(passenger_df)[0]

    return prediction, probabilities


# ================= SIDEBAR =================

with st.sidebar:

    st.markdown("# 🚢 RMS TITANIC")

    st.markdown(
        "<p class='small-caption'>THE TITANIC AI PROJECT</p>",
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Passenger Prediction",
            "Data Analytics",
            "Model Performance",
            "Passenger Explorer",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### MODEL STATUS")

    st.success("Model Connected")

    st.caption("Gradient Boosting Classifier")
    st.caption("Scikit-learn Pipeline")


# ================= DASHBOARD =================

if page == "Dashboard":

    banner(
        "TITANIC AI DASHBOARD",
        "A JOURNEY THROUGH DATA, SURVIVAL AND MACHINE LEARNING"
    )

    total = len(df)
    survived = int(df["Survived"].sum())
    lost = total - survived

    survival_rate = (
        survived / total * 100
        if total > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("TOTAL PASSENGERS", f"{total:,}")
    col2.metric("SURVIVED", f"{survived:,}")
    col3.metric("LOST", f"{lost:,}")
    col4.metric("SURVIVAL RATE", f"{survival_rate:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Survival Overview")

        survival_data = pd.DataFrame({
            "Status": ["Survived", "Did Not Survive"],
            "Passengers": [survived, lost]
        })

        fig = px.pie(
            survival_data,
            names="Status",
            values="Passengers",
            hole=0.65,
            color="Status",
            color_discrete_map={
                "Survived": "#36B6A3",
                "Did Not Survive": "#164545"
            }
        )

        fig.update_traces(
            textinfo="percent+label",
            textfont_size=13
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    with right:

        st.subheader("Survival by Passenger Class")

        class_data = (
            df.groupby("Pclass")["Survived"]
            .mean()
            .reset_index()
        )

        class_data["Survived"] = class_data["Survived"] * 100

        class_data["Pclass"] = (
            "Class " + class_data["Pclass"].astype(str)
        )

        fig = px.bar(
            class_data,
            x="Pclass",
            y="Survived",
            color="Pclass",
            color_discrete_sequence=[
                "#36B6A3",
                "#1C7770",
                "#8BD8C5"
            ],
            labels={
                "Survived": "Survival Rate (%)",
                "Pclass": "Passenger Class"
            }
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    st.divider()

    st.subheader("Quick Insights")

    c1, c2, c3 = st.columns(3)

    c1.info(f"The dataset contains {total} passenger records.")

    c2.info(
        f"{survival_rate:.1f}% of passengers in the dataset survived."
    )

    c3.info("The model uses 8 passenger features for prediction.")


# ================= PASSENGER PREDICTION =================

elif page == "Passenger Prediction":

    banner(
        "PASSENGER PREDICTION",
        "DISCOVER THE SURVIVAL PROBABILITY OF A PASSENGER"
    )

    st.write(
        "Enter passenger information below to predict the survival outcome."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            pclass = st.selectbox(
                "Passenger Class",
                [1, 2, 3]
            )

            sex = st.selectbox(
                "Gender",
                ["male", "female"]
            )

            age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=100.0,
                value=25.0
            )

            fare = st.number_input(
                "Ticket Fare",
                min_value=0.0,
                value=30.0
            )

        with col2:

            sibsp = st.number_input(
                "Siblings / Spouses",
                min_value=0,
                max_value=10,
                value=0
            )

            parch = st.number_input(
                "Parents / Children",
                min_value=0,
                max_value=10,
                value=0
            )

            embarked = st.selectbox(
                "Port of Embarkation",
                ["S", "C", "Q"]
            )

            st.caption(
                "S = Southampton | C = Cherbourg | Q = Queenstown"
            )

        family_size = sibsp + parch + 1

        st.info(f"Calculated Family Size: {family_size}")

        submitted = st.form_submit_button(
            "PREDICT SURVIVAL",
            use_container_width=True
        )

    if submitted:

        passenger = {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked,
            "FamilySize": family_size
        }

        try:

            prediction, probabilities = predict_passenger(passenger)

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                if prediction == 1:
                    st.success("PASSENGER LIKELY SURVIVED")
                else:
                    st.error("PASSENGER LIKELY DID NOT SURVIVE")

            with col2:

                st.metric(
                    "Survival Probability",
                    f"{probabilities[1] * 100:.2f}%"
                )

            probability_data = pd.DataFrame({
                "Outcome": [
                    "Did Not Survive",
                    "Survived"
                ],
                "Probability": probabilities * 100
            })

            fig = px.bar(
                probability_data,
                x="Outcome",
                y="Probability",
                color="Outcome",
                color_discrete_map={
                    "Survived": "#36B6A3",
                    "Did Not Survive": "#164545"
                },
                text_auto=".2f"
            )

            fig.update_layout(
                yaxis_title="Probability (%)",
                xaxis_title=""
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        except Exception as e:

            st.error("Prediction failed.")
            st.code(str(e))


# ================= DATA ANALYTICS =================

elif page == "Data Analytics":

    banner(
        "DATA ANALYTICS",
        "EXPLORE THE PATTERNS BEHIND TITANIC SURVIVAL"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("ROWS", df.shape[0])
    col2.metric("COLUMNS", df.shape[1])
    col3.metric(
        "MISSING VALUES",
        int(df.isnull().sum().sum())
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Age Distribution")

        fig = px.histogram(
            df,
            x="Age",
            color="Survived",
            nbins=30,
            barmode="overlay",
            color_discrete_sequence=[
                "#36B6A3",
                "#8BD8C5"
            ]
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    with right:

        st.subheader("Passengers by Gender")

        gender_data = (
            df.groupby(["Sex", "Survived"])
            .size()
            .reset_index(name="Count")
        )

        fig = px.bar(
            gender_data,
            x="Sex",
            y="Count",
            color="Survived",
            barmode="group",
            color_discrete_sequence=[
                "#164545",
                "#36B6A3"
            ]
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    left, right = st.columns(2)

    with left:

        st.subheader("Fare Distribution")

        fig = px.histogram(
            df,
            x="Fare",
            color="Survived",
            nbins=30,
            color_discrete_sequence=[
                "#164545",
                "#36B6A3"
            ]
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    with right:

        st.subheader("Survival by Embarkation")

        embarked_data = (
            df.groupby("Embarked")["Survived"]
            .mean()
            .reset_index()
        )

        embarked_data["Survived"] *= 100

        fig = px.bar(
            embarked_data,
            x="Embarked",
            y="Survived",
            color="Embarked",
            color_discrete_sequence=[
                "#36B6A3",
                "#1C7770",
                "#8BD8C5"
            ],
            labels={
                "Survived": "Survival Rate (%)"
            }
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    st.divider()

    st.subheader("Correlation Heatmap")

    numeric_data = df.select_dtypes(include=np.number)

    correlation = numeric_data.corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale=[
            "#061B1B",
            "#164545",
            "#36B6A3",
            "#D8E8E2"
        ],
        aspect="auto"
    )

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True
    )


# ================= MODEL PERFORMANCE =================

elif page == "Model Performance":

    banner(
        "MODEL PERFORMANCE",
        "EVALUATING THE MACHINE LEARNING MODEL"
    )

    st.write(
        "The saved Gradient Boosting model is evaluated on the test split."
    )

    try:

        X = df[FEATURES]
        y = df["Survived"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        col1, col2 = st.columns(2)

        col1.metric(
            "TEST ACCURACY",
            f"{accuracy * 100:.2f}%"
        )

        col2.metric(
            "TEST SAMPLES",
            len(y_test)
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig = px.imshow(
                cm,
                text_auto=True,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                color_continuous_scale=[
                    "#061B1B",
                    "#164545",
                    "#36B6A3"
                ],
                aspect="auto"
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

        with right:

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(3),
                use_container_width=True
            )

        st.divider()

        st.subheader("Model Information")

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Algorithm:** Gradient Boosting Classifier")
            st.write("**Problem Type:** Binary Classification")
            st.write("**Target:** Survived")

        with col2:

            st.write("**Train-Test Split:** 80% / 20%")
            st.write("**Random State:** 42")
            st.write("**Input Features:** 8")

    except Exception as e:

        st.error("Could not calculate model performance.")
        st.code(str(e))


# ================= PASSENGER EXPLORER =================

elif page == "Passenger Explorer":

    banner(
        "PASSENGER EXPLORER",
        "EXPLORE PASSENGER RECORDS FROM THE TITANIC"
    )

    st.subheader("Filter Passengers")

    col1, col2, col3 = st.columns(3)

    with col1:

        classes = sorted(df["Pclass"].dropna().unique())

        selected_class = st.multiselect(
            "Passenger Class",
            classes,
            default=classes
        )

    with col2:

        genders = list(df["Sex"].dropna().unique())

        selected_gender = st.multiselect(
            "Gender",
            genders,
            default=genders
        )

    with col3:

        selected_survival = st.multiselect(
            "Survival",
            [0, 1],
            default=[0, 1],
            format_func=lambda x: (
                "Survived" if x == 1 else "Did Not Survive"
            )
        )

    filtered_df = df[
        (df["Pclass"].isin(selected_class)) &
        (df["Sex"].isin(selected_gender)) &
        (df["Survived"].isin(selected_survival))
    ]

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric(
        "MATCHING PASSENGERS",
        len(filtered_df)
    )

    if len(filtered_df) > 0:

        rate = filtered_df["Survived"].mean() * 100

    else:

        rate = 0

    col2.metric(
        "SURVIVAL RATE",
        f"{rate:.1f}%"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "DOWNLOAD FILTERED DATA",
        data=csv,
        file_name="titanic_filtered_passengers.csv",
        mime="text/csv"
    )


# ================= ABOUT PROJECT =================

elif page == "About Project":

    banner(
        "ABOUT THE PROJECT",
        "MACHINE LEARNING • DATA ANALYSIS • PREDICTION"
    )

    st.subheader("Titanic Survival Prediction")

    st.write("""
    This project uses machine learning to predict whether a passenger
    would have survived the Titanic disaster.

    The model learns patterns from passenger information such as
    passenger class, gender, age, family size, fare and port of embarkation.
    """)

    st.divider()

    st.subheader("Project Workflow")

    steps = [
        "Dataset Collection",
        "Data Cleaning",
        "Exploratory Data Analysis",
        "Missing Value Handling",
        "Feature Engineering",
        "One-Hot Encoding",
        "Feature Scaling",
        "Model Training",
        "Model Evaluation",
        "Survival Prediction"
    ]

    for number, step in enumerate(steps, 1):

        st.markdown(f"**{number}.** {step}")

    st.divider()

    st.subheader("Machine Learning Details")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Algorithm**")
        st.write("Gradient Boosting Classifier")

        st.write("**Problem Type**")
        st.write("Binary Classification")

        st.write("**Target Column**")
        st.write("Survived")

    with col2:

        st.write("**Dataset**")
        st.write("Titanic Passenger Dataset")

        st.write("**Input Features**")
        st.write("8 passenger features")

        st.write("**Framework**")
        st.write("Scikit-learn and Streamlit")

    st.divider()

    st.markdown("""
    <p style="text-align:center; color:#63D5BE; letter-spacing:2px;">
        🚢 RMS TITANIC • TITANIC AI PROJECT
    </p>
    """, unsafe_allow_html=True)
