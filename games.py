import streamlit as st
import json
import pandas as pd
import plotly.express as px
import google.generativeai as genai

genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

model = genai.GenerativeModel("gemini-flash-latest")

st.set_page_config(page_title="Toy Store Sales Dashboard", layout="wide")

st.title("🧸 AI Toy Store Sales Dashboard For Toy Store Sales")
st.markdown("Interactive Dashboard using Streamlit, Pandas , Genai, Json and Plotly")
# -----------------------
# Load CSV Files
# -----------------------
sales = pd.read_csv("sales.csv")
products = pd.read_csv("products.csv")
stores = pd.read_csv("stores.csv")
inventory = pd.read_csv("inventory.csv")

# -----------------------
# Merge Data
# -----------------------
df = pd.merge(sales, products, on="Product_ID", how="left")
df = pd.merge(df, stores, on="Store_ID", how="left")
df = pd.merge(df, inventory, on=["Store_ID", "Product_ID"], how="left")

# -----------------------
# Create New Columns
# -----------------------
df["Revenue"] = df["Units"] * df["Product_Price"]
df["Profit"] = df["Units"] * (df["Product_Price"] - df["Product_Cost"])

df["Date"] = pd.to_datetime(df["Date"])

# -----------------------
# Sidebar Filters
# -----------------------
st.sidebar.header("Filters")

category = st.sidebar.multiselect(
    "Select Category",
    df["Product_Category"].unique(),
    default=df["Product_Category"].unique()
)

city = st.sidebar.multiselect(
    "Select City",
    df["Store_City"].unique(),
    default=df["Store_City"].unique()
)

filtered = df[
    (df["Product_Category"].isin(category)) &
    (df["Store_City"].isin(city))
]

# -----------------------
# KPI Cards
# -----------------------
total_sales = filtered["Units"].sum()
total_revenue = filtered["Revenue"].sum()
total_profit = filtered["Profit"].sum()
products_count = filtered["Product_Name"].nunique()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Units Sold", f"{total_sales:,}")
c2.metric("Total Revenue", f"${total_revenue:,.2f}")
c3.metric("Total Profit", f"${total_profit:,.2f}")
c4.metric("Products", products_count)

st.divider()

# -----------------------
# Revenue by Category
# -----------------------
cat = filtered.groupby("Product_Category")["Revenue"].sum().reset_index()

fig = px.bar(
    cat,
    x="Product_Category",
    y="Revenue",
    color="Product_Category",
    title="Revenue by Product Category"
)

st.plotly_chart(fig, use_container_width=True)
compare = filtered.groupby("Product_Category")[["Revenue","Profit"]].sum().reset_index()

fig = px.bar(
    compare,
    x="Product_Category",
    y=["Revenue","Profit"],
    barmode="group",
    title="Revenue vs Profit by Category"
)

st.plotly_chart(fig, width="stretch")

# -----------------------
# Top Products
# -----------------------
top = filtered.groupby("Product_Name")["Revenue"].sum().nlargest(10).reset_index()

fig = px.bar(
    top,
    x="Revenue",
    y="Product_Name",
    orientation="h",
    color="Revenue",
    title="Top 10 Products"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Revenue by City
# -----------------------
city_sales = filtered.groupby("Store_City")["Revenue"].sum().reset_index()

fig = px.pie(
    city_sales,
    values="Revenue",
    names="Store_City",
    title="Revenue by City"
)
st.plotly_chart(fig, width="stretch")

# -----------------------
# Monthly Revenue Trend
# -----------------------
monthly = filtered.groupby(filtered["Date"].dt.to_period("M"))["Revenue"].sum().reset_index()

monthly["Date"] = monthly["Date"].astype(str)

fig = px.line(
    monthly,
    x="Date",
    y="Revenue",
    markers=True,
    title="Monthly Revenue Trend"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Inventory
# -----------------------
stock = filtered.groupby("Product_Category")["Stock_On_Hand"].sum().reset_index()

fig = px.bar(
    stock,
    x="Product_Category",
    y="Stock_On_Hand",
    color="Stock_On_Hand",
    title="Inventory by Category"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Data Table
# -----------------------
st.subheader("Merged Dataset")
st.dataframe(filtered)

# -----------------------
# AI Dashboard Generator
# -----------------------

st.divider()
st.subheader("🤖 AI Dashboard Generator")

user_prompt = st.text_area(
    "What would you like to visualize?",
    placeholder="Example: Show me the top 5 products by profit"
)

if st.button("Generate Dashboard", key="ai_button"):

    prompt = f"""
You are a Business Intelligence Dashboard Planner.

The user will describe what they want to see from a toy store sales dataset.

AVAILABLE COLUMNS:

Dimensions:
- Product_Category
- Product_Name
- Store_City
- Date

Metrics:
- Revenue
- Profit
- Units
- Stock_On_Hand

AVAILABLE CHART TYPES:
- bar
- line
- pie

USER REQUEST:
{user_prompt}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "charts": [
        {{
            "type": "bar",
            "group_by": "Product_Category",
            "metric": "Revenue",
            "aggregation": "sum",
            "top_n": null,
            "title": "Revenue by Product Category"
        }}
    ]
}}

RULES:

1. Only use columns listed above.
2. Only use bar, line, or pie charts.
3. aggregation must be "sum" or "mean".
4. top_n must be an integer or null.
5. Generate between 1 and 4 charts.
6. Do not explain anything.
7. Return JSON only.
"""

    try: 

        response = model.generate_content(prompt)

        # Get Gemini response
        ai_text = response.text.strip()

        # Remove markdown code fences if Gemini adds them
        ai_text = ai_text.replace("```json", "")
        ai_text = ai_text.replace("```", "")
        ai_text = ai_text.strip()

        # Convert JSON text into Python dictionary
        dashboard = json.loads(ai_text)

        st.subheader("🤖 AI Generated Dashboard")

        # Create every chart requested by Gemini
        for chart in dashboard["charts"]:

            chart_type = chart["type"]
            group_by = chart["group_by"]
            metric = chart["metric"]
            aggregation = chart["aggregation"]
            top_n = chart["top_n"]
            title = chart["title"]

            # -----------------------
            # Prepare data
            # -----------------------

            chart_df = filtered.copy()

            # Special handling for Date
            if group_by == "Date":

                chart_df["Month"] = (
                    chart_df["Date"]
                    .dt.to_period("M")
                    .astype(str)
                )

                group_column = "Month"

            else:

                group_column = group_by

            # -----------------------
            # Aggregation
            # -----------------------

            if aggregation == "sum":

                chart_data = (
                    chart_df
                    .groupby(group_column)[metric]
                    .sum()
                    .reset_index()
                )

            elif aggregation == "mean":

                chart_data = (
                    chart_df
                    .groupby(group_column)[metric]
                    .mean()
                    .reset_index()
                )

            # -----------------------
            # Top N
            # -----------------------

            if top_n is not None:

                chart_data = (
                    chart_data
                    .nlargest(top_n, metric)
                )

            # -----------------------
            # Create Chart
            # -----------------------

            if chart_type == "bar":

                fig = px.bar(
                    chart_data,
                    x=group_column,
                    y=metric,
                    title=title
                )

            elif chart_type == "line":

                fig = px.line(
                    chart_data,
                    x=group_column,
                    y=metric,
                    markers=True,
                    title=title
                )

            elif chart_type == "pie":

                fig = px.pie(
                    chart_data,
                    names=group_column,
                    values=metric,
                    title=title
                )

            # -----------------------
            # Display Chart
            # -----------------------

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    except Exception as e:

        st.error(f"Could not generate dashboard: {e}")

