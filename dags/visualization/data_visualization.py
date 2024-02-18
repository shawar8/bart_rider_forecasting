import streamlit as st
import pandas as pd
from datetime import timedelta
import plotly.express as px

def plot_line_plot(df: pd.DataFrame, pred_start_date):
    df.index.set_names(['date'], inplace=True)
    df.reset_index(inplace=True)
    df.loc[df['date'] < pred_start_date, 'data_type'] = 'Observed'
    df.loc[df['date'] >= pred_start_date, 'data_type'] = 'Predicted'
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df['num_passengers'] = df['num_passengers'].astype(int)
    fig = px.line(df, x='date', y='num_passengers', color='data_type', markers=True,
                  labels = {'date': 'Date',
                            'num_passengers': 'Number of Passengers',
                            'data_type': 'Date Type'},
                  title= 'Daily Number of BART Passengers')

    st.plotly_chart(fig, use_container_width=True)
    add_checkbox = st.sidebar.checkbox('Show DataFrame')
    if add_checkbox:
        st.table(df)
