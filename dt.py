import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import base64
#from dtreeviz.trees import dtreeviz
import psycopg2
from sqlalchemy import create_engine
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from sqlalchemy import Table, Column, String, MetaData
from datetime import datetime

def insert_row(session_id,engine):
    if engine.execute("SELECT session_id FROM session_state WHERE session_id = '%s'" % (session_id)).fetchone() is None:
            engine.execute("""INSERT INTO session_state (session_id) VALUES ('%s')""" % (session_id))

def update_row(column,new_value,session_id,engine):
    engine.execute("UPDATE session_state SET %s = '%s' WHERE session_id = '%s'" % (column,new_value,session_id))

def get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id

    return session_info.session, session_id

def confusion_matrix_plot(data,labels):
    z = data.tolist()[::-1]
    x = labels
    y = labels
    z_text = z

    fig = ff.create_annotated_heatmap(z, x, y, annotation_text=z_text, text=z,hoverinfo='text',colorscale='Blackbody')
    fig.update_layout(font_family="IBM Plex Sans")

    st.write(fig)

def roc_plot(data):
    fig = px.line(data, x="False Positive", y="True Positive")#, title='ROC Curve')
    fig.update_layout(font_family="IBM Plex Sans")

    st.write(fig)

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

def dt_viz(X,y,label_name,model,feature_cols,class_names):
    class_names = [str(x) for x in class_names]
    viz = dtreeviz(model, X, y, target_name=label_name,feature_names=feature_cols, class_names=class_names)
    st.image(viz._repr_svg_(), use_column_width=True)

def dt_main():

    #df = pd.read_csv('C:/Users/Mohammad Khorasani/Desktop/data.csv')
    #st.sidebar.title('Decision Tree Classifer')
    st.sidebar.subheader('Training Dataset')
    status, df = file_upload('Please upload a training dataset')

    #engine = create_engine('''postgres://aypucbrafyqczq:da4c68db377bf354ea19986448fd55d59b7c7cbb08aba696ed8c2bd293283174@ec2-54-211-160-34.compute-1.amazonaws.com:5432/df4hngkj04sb9t''')
    DATABASE_URL = os.environ['DATABASE_URL']
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    engine = conn.cursor()

    _, session_id = get_session()

    insert_row(session_id,engine)
    update_row('dt1',datetime.now().strftime("%H:%M:%S %d/%m/%Y"),session_id,engine)

    if status == True:
        update_row('data1_rows',len(df),session_id,engine)
        update_row('dt2',datetime.now().strftime("%H:%M:%S %d/%m/%Y"),session_id,engine)
        col_names = list(df)

        st.title('Training')
        st.subheader('Parameters')
        col1, col2, col3 = st.beta_columns((3,3,2))

        with col1:
            feature_cols = st.multiselect('Please select features',col_names)
        with col2:
            label_col = st.selectbox('Please select label',col_names)
        with col3:
            test_size = st.number_input('Please enter test size',0.01,0.99,0.25,0.05)

        with st.beta_expander('Advanced Parameters'):
            col2_1, col2_2 = st.beta_columns(2)
            with col2_1:
                criterion = st.selectbox('Criterion',['gini','entropy'])
                max_depth = st.radio('Max depth',[None,'Custom'])
                if max_depth == 'Custom':
                    max_depth = st.number_input('Custom max depth',0,100,1,1)
                min_samples_leaf = st.number_input('Min samples leaf',0,99,1,1)
                max_features = st.selectbox('Max features',[None,'auto','sqrt','log2','Custom'])
                if max_features == 'Custom':
                    max_features = st.number_input('Custom max features',0.0,99.0,1.0,0.1)
                max_leaf_nodes = st.radio('Max leaf nodes',[None,'Custom'])
                if max_leaf_nodes == 'Custom':
                    max_leaf_nodes = st.number_input('Custom max leaf nodes',2,99,2,1)
                min_impurity_split = st.number_input('Min impurity split',0,99,0,1)
            with col2_2:
                splitter = st.selectbox('Splitter',['best','random'])
                min_samples_split = st.number_input('Min samples split',2,99,2,1)
                min_weight_fraction_leaf = st.number_input('Min weight fraction leaf',0.0,99.0,0.0,0.1)
                random_state = st.radio('Random state',[None,'Custom'])
                if random_state == 'Custom':
                    random_state = st.number_input('Custom random state',0,99,1,1)
                min_impurity_decrease = st.number_input('Min impurity decrease',0.0,99.0,0.0,0.1)
                class_weight = st.radio('Class weight',[None,'balanced'])
                ccp_alpha = st.number_input('Complexity parameter',0.0,99.0,0.0,0.1)

            st.markdown('For further information please refer to ths [link](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)')

        try:
            X = df[feature_cols]
            y = df[label_col]
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=0)
            clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                                         max_features=max_features, random_state=random_state,
                                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                         min_impurity_split=min_impurity_split, class_weight=class_weight,
                                         ccp_alpha=ccp_alpha)
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

            st.subheader('Confusion Matrix')
            confusion_matrix_plot(cnf_matrix,list(df[label_col].unique()))

            #Decision tree viz
            #try:
            #    dt_viz(X_train,y_train,label_col,clf,feature_cols,list(df[label_col].unique()))
            #except:
            #    pass

            accuracy = metrics.accuracy_score(y_test, y_pred)
            st.subheader('Metrics')
            st.info('Accuracy: **%s**' % (round(accuracy,3)))

            try:
                y_pred_proba = clf.predict_proba(X_test)[::,1]
                fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
                roc_data = pd.DataFrame([])
                roc_data['True Positive'] =  tpr
                roc_data['False Positive'] = fpr
                st.subheader('ROC Curve')
                roc_plot(roc_data)
                auc = metrics.roc_auc_score(y_test, y_pred_proba)
                st.info('Area Under Curve: **%s**' % (round(auc,3)))
            except:
                pass

            st.sidebar.subheader('Test Dataset')
            status_test, df_test = file_upload('Please upload a test dataset')
            update_row('dt3',datetime.now().strftime("%H:%M:%S %d/%m/%Y"),session_id,engine)

            if status_test == True:
                try:
                    st.title('Testing')
                    update_row('data2_rows',len(df_test),session_id,engine)
                    X_test_test = df_test[feature_cols]
                    y_pred_test = clf.predict(X_test_test)

                    X_pred = df_test.copy()
                    X_pred[label_col] = y_pred_test
                    X_pred = X_pred.sort_index()

                    st.subheader('Predicted Labels')
                    st.write(X_pred)
                    #st.write(X_pred[label_col].value_counts())
                    st.markdown(download(X_pred,'DummyLearn.com - Decision Tree Classifier - Predicted Labels'), unsafe_allow_html=True)
                    update_row('dt4',datetime.now().strftime("%H:%M:%S %d/%m/%Y"),session_id,engine)
                except:
                    st.warning('Please upload a test dataset with the same feature set as the training dataset')

            elif status_test == False:
                st.sidebar.warning('Please upload a test dataset')

        except:
            st.warning('Please select at least one feature, a suitable label and appropriate advanced paramters')

    elif status == False:
        st.title('Welcome 🌱')
        st.subheader('Please use the left pane to upload your dataset')
        st.sidebar.warning('Please upload a training dataset')

    st.sidebar.subheader('Sample Dataset')
    if st.sidebar.button('Download sample dataset'):
        url = 'https://raw.githubusercontent.com/mkhorasani/dummylearn/main/Sample%20datasets/data2.csv'
        csv = pd.read_csv(url)
        st.sidebar.markdown(download(csv,'sample_dataset'), unsafe_allow_html=True)
