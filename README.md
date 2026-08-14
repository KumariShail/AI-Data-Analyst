## 🧸 AI Toy Sales Dashboard


An interactive toy store sales analytics dashboard built using **Python, Pandas, Streamlit, Plotly, and Google Gemini**.


The project combines a traditional business intelligence dashboard with an **AI-powered dashboard generator**. Users can explore predefined sales insights through interactive filters and KPI cards, or describe what they want to visualize and let Gemini generate a visualization plan dynamically.


---


## ✨ Features


### 📊 Interactive Sales Dashboard


- Total Units Sold
- Total Revenue
- Total Profit
- Number of Products
- Revenue by Product Category
- Revenue vs Profit by Category
- Top 10 Products
- Revenue by City
- Monthly Revenue Trend
- Inventory by Category
- Interactive category and city filters
- Merged sales, product, store, and inventory data


### 🤖 AI Dashboard Generator


The application includes an AI-powered dashboard generator using **Google Gemini**.


Users can enter a natural-language request such as:


> **"Show me the top 5 products by profit"**


Gemini analyzes the request and returns a structured JSON visualization plan containing:


- Chart type
- Grouping column
- Metric
- Aggregation
- Top N
- Chart title


The Python application then converts this configuration into interactive **Plotly charts**.


---


## 🧠 How It Works


```text
Sales + Product + Store + Inventory Data
                    ↓
              Pandas Processing
                    ↓
           Data Cleaning & Merging
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
   Normal Dashboard       Gemini AI
          ↓                   ↓
     Plotly Charts       JSON Chart Plan
                              ↓
                         Plotly Charts

The application uses Pandas for data processing and Gemini for generating the visualization plan. Plotly then renders the charts dynamically inside Streamlit.

🛠️ Technologies
Python — Core programming language
Pandas — Data processing and analysis
Streamlit — Interactive web application
Plotly — Interactive data visualizations
Google Gemini API — AI-powered visualization planning
JSON — Structured communication between Gemini and the application
📂 Dataset

The dashboard uses four datasets:

sales.csv
products.csv
stores.csv
inventory.csv

The datasets are merged using product and store identifiers before analysis.

Additional metrics such as Revenue and Profit are calculated during data processing.

🤖 AI Dashboard Example

Users can enter natural-language requests such as:

"Show me the top 5 products by profit"

Gemini generates a visualization configuration based on the request.

The application then processes the configuration and dynamically creates the requested Plotly charts.

This allows users to interact with the data using natural language instead of manually selecting every visualization.

🔮 Future Improvements

The current version uses a predefined dataset schema when communicating with Gemini.

Future versions will aim to:

Automatically inspect uploaded datasets
Detect numerical, categorical, and date columns
Automatically generate dataset summaries
Allow users to upload different CSV datasets
Let AI select appropriate KPIs and visualizations
Generate dashboards without hard-coded column names
Support dynamic dashboard generation for previously unseen datasets
👩‍💻 Author

Kumari Shail

