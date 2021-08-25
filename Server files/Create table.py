import psycopg2
from sqlalchemy import create_engine

engine = create_engine('''postgres://aypucbrafyqczq:da4c68db377bf354ea19986448fd55d59b7c7cbb08aba696ed8c2bd293283174@ec2-54-211-160-34.compute-1.amazonaws.com:5432/df4hngkj04sb9t''')

def create_table(engine):
    engine.execute("""CREATE TABLE IF NOT EXISTS session_state (session_id text PRIMARY KEY,lr1 text,lr2 text,lr3 text,lr4 text,
                            nb1 text,nb2 text,nb3 text,nb4 text,dt1 text,dt2 text,dt3 text,dt4 text,knn1 text,knn2 text,
                            knn3 text,knn4 text,svm1 text,svm2 text,svm3 text,svm4 text,data1_rows text,data2_rows text)""")

def insert_row(session_id,engine):
    if engine.execute("SELECT session_id FROM session_state WHERE session_id = '%s'" % (session_id)).fetchone() is None:
            engine.execute("""INSERT INTO session_state (session_id) VALUES ('%s')""" % (session_id))

def update_row(column,new_value,session_id,engine):
    engine.execute("UPDATE session_state SET %s = '%s' WHERE session_id = '%s'" % (column,new_value,session_id))
