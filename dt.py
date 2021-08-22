import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import base64
from dtreeviz.trees import dtreeviz

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
                st.error('Unable to save file. Please ensure file is .csv or .xlsx format.')
                return content, None
    else:
        return content, None

def dt_viz(X,y,label_name,model,feature_cols,class_names):
    class_names = [str(x) for x in class_names]
    viz = dtreeviz(model, X, y, target_name=label_name,feature_names=feature_cols, class_names=class_names)
    st.image(viz._repr_svg_(), use_column_width=True)

if __name__ == '__main__':

    st.set_page_config(
        layout="centered",
        initial_sidebar_state="expanded",
        page_title='DummyLearn.com',
    )

    hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    """
    st.markdown(hide_footer_style, unsafe_allow_html=True)

    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    #df = pd.read_csv('C:/Users/Mohammad Khorasani/Desktop/data.csv')
    st.sidebar.title('Decision Tree Classifer')
    st.sidebar.subheader('Training Dataset')
    status, df = file_upload('Please upload a training dataset')

    if status == True:
        col_names = list(df)

        st.title('Training')
        st.subheader('Parameters')
        col1, col2 = st.beta_columns((2,1))
        col1, col2, col3 = st.beta_columns((3,3,2))

        with col1:
            feature_cols = st.multiselect('Please select features',col_names)
        with col2:
            label_col = st.selectbox('Please select label',col_names)
        with col3:
            test_size = st.number_input('Please enter test size',0.01,0.99,0.25,0.05)

        try:
            X = df[feature_cols]
            y = df[label_col]
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=0)
            clf = DecisionTreeClassifier()
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

            st.subheader('Confusion Matrix')
            confusion_matrix_plot(cnf_matrix,list(df[label_col].unique()))

            try:
                dt_viz(X_train,y_train,label_col,clf,feature_cols,list(df[label_col].unique()))
            except:
                pass

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

            if status_test == True:
                try:
                    st.title('Testing')
                    X_test_test = df_test[feature_cols]
                    y_pred_test = clf.predict(X_test_test)

                    X_pred = df_test.copy()
                    X_pred[label_col] = y_pred_test
                    X_pred = X_pred.sort_index()

                    st.subheader('Predicted Labels')
                    st.write(X_pred)
                    #st.write(X_pred[label_col].value_counts())
                    st.markdown(download(X_pred,'DummyLearn.com - Decision Tree Classifier - Predicted Labels'), unsafe_allow_html=True)
                except:
                    st.warning('Please upload a test dataset with the same feature set as the training dataset')

            elif status_test == False:
                st.sidebar.warning('Please upload a test dataset')

        except:
            st.warning('Please select at least one feature and a suitable label')

    elif status == False:
        st.sidebar.warning('Please upload a training dataset')
