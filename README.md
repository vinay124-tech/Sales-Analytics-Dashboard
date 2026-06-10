Antigravity Sales Command Center

This is a full-stack data engineering and analytics dashboard I built from scratch. The goal of this project was to practice building an end-to-end data pipeline and integrating LLMs for automated data analysis, without relying on pre-cleaned Kaggle datasets.

I went with an "antigravity tech" theme for the dummy data just to make it more interesting than standard retail data.

Tech Stack

Data Generation & Cleaning: Python, Pandas, Numpy, Faker

Analytics / OLAP: DuckDB (runs directly on local CSVs)

Frontend Dashboard: Streamlit, Plotly

AI Agent: Google GenAI SDK (Gemini 2.5 Flash)

Project Architecture

The project is broken down into a sequential pipeline:

generate_data.py - Synthesizes 12 weeks of raw, messy sales data. Deliberately introduces duplicates, negative quantities, and extreme price outliers.

clean_data.py - A Pandas script that deduplicates rows, handles the negative values, and uses the IQR (Interquartile Range) method to detect and cap price outliers. Outputs cleaned_*.csv files.

analytics.py / insight_engine.py - Backend logic testing DuckDB queries to calculate WoW revenue growth, AOV, and funnel drop-offs, then passing the results to the Gemini API for natural language insights.

app.py - The main Streamlit application that ties the SQL queries, UI filters, charts, and AI insight generation together.

How to Run It Locally

Install dependencies:

pip install -r requirements.txt


Generate and clean the data:
You need to build the local database files first.

python generate_data.py
python clean_data.py


Set your API Key:
The AI insight engine requires a free Google Gemini API key. Set it in your terminal before running the app.

Mac/Linux: export GEMINI_API_KEY="your_api_key_here"

Windows CMD: set GEMINI_API_KEY="your_api_key_here"

Launch the app:

streamlit run app.py


Known Issues / Next Steps

Right now, DuckDB runs queries directly against the CSVs. It's fast enough for 5,000 rows, but if I scale the data generation script up, I should probably convert the storage format to Parquet.

The UI filters trigger a re-run of the whole Streamlit script. I might implement @st.cache_data on the DuckDB connection later to optimize the load times.
