import psycopg2
from sqlalchemy import create_engine

engine = create_engine('''postgres://jtslpiqkuneekd:5d0a8c1b83cee260efde77bbfb0fb41b13dfb0e4fde4443ee8be6e0bfa2ecee3@ec2-35-153-114-74.compute-1.amazonaws.com:5432/d9en3c9i44m7h9''')

def create_table(engine):
    engine.execute("""CREATE TABLE IF NOT EXISTS session_state (session_id text PRIMARY KEY,lr1 text,lr2 text,lr3 text,lr4 text,
                            nb1 text,nb2 text,nb3 text,nb4 text,dt1 text,dt2 text,dt3 text,dt4 text,knn1 text,knn2 text,
                            knn3 text,knn4 text,svm1 text,svm2 text,svm3 text,svm4 text,data1_rows text,data2_rows text)""")

def insert_row(session_id,engine):
    if engine.execute("SELECT session_id FROM session_state WHERE session_id = '%s'" % (session_id)).fetchone() is None:
            engine.execute("""INSERT INTO session_state (session_id) VALUES ('%s')""" % (session_id))

def update_row(column,new_value,session_id,engine):
    engine.execute("UPDATE session_state SET %s = '%s' WHERE session_id = '%s'" % (column,new_value,session_id))
