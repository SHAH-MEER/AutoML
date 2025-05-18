import streamlit as st

st.set_page_config(
    page_title="AutoML Web App",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    """
    <style>
    .social-container {
        display: flex;
        justify-content: flex-end;
        gap: 15px;
        padding: 10px;
    }
    .social-icon {
        font-size: 24px;
        color: white;
    }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <div class="social-container">
        <a href="https://github.com/SHAH-MEER" target="_blank">
            <i class="fab fa-github social-icon"></i>
        </a>
        <a href="https://www.linkedin.com/in/shahmeer-shahzad-790b67356/" target="_blank">
            <i class="fab fa-linkedin social-icon"></i>
        </a>
        <a href="mailto:shahmeershahzad67@gmail.com" target="_blank">
            <i class="fas fa-envelope social-icon"></i>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🤖 AutoML Web App")
st.markdown("""
Welcome to AutoML Web App - Your one-stop solution for automated machine learning!

This application allows you to perform automated classification and regression tasks with ease.

Navigate to the specific task pages using the sidebar on the left to:

- **Classification Tasks** 📊
- **Regression Tasks** 📈

Each task page provides options to load datasets, configure models, run analysis, and visualize results.

For more details on setup and usage, please refer to the `README.md` file.
""")

st.sidebar.success("Select a task type from above.")