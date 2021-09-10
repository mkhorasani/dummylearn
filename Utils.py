import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, String, MetaData
import psycopg2
import streamlit as st
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
import os
import base64
from datetime import datetime

def insert_row(session_id,engine):
    if engine.execute("SELECT session_id FROM session_state WHERE session_id = '%s'" % (session_id)).fetchone() is None:
            engine.execute("""INSERT INTO session_state (session_id) VALUES ('%s')""" % (session_id))

def update_row(column,new_value,session_id,engine):
    if engine.execute("SELECT %s FROM session_state WHERE session_id = '%s'" % (column,session_id)).first()[0] is None:
        engine.execute("UPDATE session_state SET %s = '%s' WHERE session_id = '%s'" % (column,new_value,session_id))

def get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id

    return session_info.session, session_id

def download(df,filename): # Downloading DataFrame
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = (f'<a href="data:file/csv;base64,{b64}" download="%s.csv">Download csv file</a>' % (filename))
    return href

def file_upload(name):
    uploaded_file = st.sidebar.file_uploader('%s' % (name),key='%s' % (name),accept_multiple_files=False)
    content = False
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            content = True
            return content, uploaded_df
        except:
            try:
                uploaded_df = pd.read_excel(uploaded_file)
                content = True
                return content, uploaded_df
            except:
                st.error('Please ensure file is .csv or .xlsx format and/or reupload file')
                return content, None
    else:
        return content, None
