# from datetime import timedelta

import sys
sys.path.append('/var/lib/airflow/airflow')
sys.path.append('/var/lib/airflow/airflow/func/connection')

import ast
from pyparsing import null_debug_action
from datetime import datetime

from airflow import DAG
import airflow.utils.dates
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator # python operator
from airflow.models import Variable
import pandas as pd
import pymysql

from airflow.decorators import dag,task
from airflow.utils.dates import days_ago
from airflow.operators.python import get_current_context
from collections import OrderedDict, defaultdict
import pymssql
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit

import json
import numpy as np

default_args = {'owner': 'airflow',}

# DB Connection 함수 불러오기
from db_conn import get_mssql_conn


def stringtojson(request):
    
    '''
    string을 json으로 변경하는 함수입니다. 
    :param request: request(string)
    :return: req_dict(json,dict)
    """

    '''

    if isinstance(request, str): 
        
        req_dict = dict()
        model_lst_pre = ast.literal_eval(request) # dict로 변경되었으나, value값은 string!
        
        for key_name in model_lst_pre.keys():
            
            if '[' in model_lst_pre[key_name]: # 여러값이 존재하는 경우
                req_dict[key_name] = [n.strip() for n in model_lst_pre[key_name].strip().replace('[','').replace(']','').split(',')]
            else:
                req_dict[key_name] = model_lst_pre[key_name]
    

        # req_dict2 = req_dict.copy()
        return req_dict


def fit_transform(X: pd.DataFrame):
    encoder_dict = defaultdict(LabelEncoder)            
    output = X.copy()
    output = X.apply(lambda x:encoder_dict[x.name].fit_transform(x))
    return output

@task(multiple_outputs = True)
def seg_extract():
    context = get_current_context()
    #request 값을 두개를 받음(1. without segcon, 2. segcon)
    #request 값이 dict 상태로 들어옴 -> stringtojson 함수 불필요
    req = context["params"]["request2"]["segcon"]
    request = context["params"]["request1"]

    if request['iutype'] == 'IR':
        if req == '':
            query = '''SELECT * FROM 아이템 WHERE 1=1'''
        else:
            query = '''SELECT * FROM 아이템 WHERE 1=1 {}'''.format(req)  # 아이템 세그먼테이션
    else:
        if req == '':
            query = '''SELECT * FROM 유저 WHERE 1=1'''
        else:
            query = '''SELECT * FROM 유저 WHERE 1=1 {}'''.format(req)  # 유저 세그먼테이션

    var_lst = []
    valist = request['varlist']
    # print('-----------------here!')
    # print(valist)
    valist = valist[1:len(valist) - 1]
    var_lst = valist.split(', ')

    # DB Connection
    with get_mssql_conn() as conn:
        
        # 데이터 불러오기
        data = pd.read_sql(sql=query, con=conn)

        # 변수 선택
        if request['iutype'] == 'IR':
            data = data[var_lst]
            data_id = data[['ITEM_CD']]
            data = data[data.columns.difference(['ITEM_CD'])]        

        else:
            data = data[var_lst]
            data_id = data[['USER_CD']]
            data = data[data.columns.difference(['S_CD','USER_CD'])]

        categorical_cols = [] # 카테고리형 
        numeric_cols = []  # 수치형 

        df = data.copy()

        # 결측치 있는 컬럼 삭제하는 for문 돌려보기
        #카데고리형 데이터 처리
        for col in df.columns:
            if df[col].dtypes == 'object': # 카데고리형 데이터 분류
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        try:
            cat_df = fit_transform(df[categorical_cols]) # 라벨인코딩
            num_df = df[numeric_cols]
            total_df = pd.concat([num_df,cat_df],axis=1)

        except ValueError:
            total_df = num_df.copy()

        # 결측치 처리 70프로 이상이면 컬럼 삭제 , 70프로 미만일 경우 행(row) 삭제
        for col in total_df.columns:
            if total_df[col].isnull().sum()/len(data)*100 >= 70:
                del total_df[col]
            elif 0 < total_df[col].isnull().sum()/len(data)*100 < 70:
                total_df = total_df.dropna(axis=0)
            else:
                total_df = total_df

        # Nan 값 처리 
        total_df = total_df.dropna(axis=0)
        
        # null 값 처리후, data_id 데이터프레임과 row 건수가 맞지 않는 경우
        data_id = data_id.iloc[total_df.index]

        min_max_scaler = MinMaxScaler()
        sc_output = min_max_scaler.fit_transform(total_df)
        df_sc = pd.DataFrame(sc_output, columns=total_df.columns, index=list(total_df.index.values))


        #KMEANS CLUSTERING
        sil = []
        kmax = 10
        segcnt = request['segcnt']
        cnt = int(segcnt)

        for k in range(2,kmax+1):
            kmeans = KMeans(n_clusters=k).fit(df_sc)
            labels = kmeans.labels_
            sil.append(silhouette_score(df_sc,labels,metric='euclidean'))

        if (cnt == 1):
            optimize_k = sil.index(max(sil)) + 2
        else:
            optimize_k = cnt;

        kmeansmodel = KMeans(n_clusters= optimize_k, init='k-means++', random_state=0)
        y_kmeans_fit = kmeansmodel.fit_predict(df_sc)
        
        df_sc['SEG_NO'] = y_kmeans_fit
        df_sc = pd.concat([df_sc, data_id],axis=1)

        # 전체 클러스터 db 저장 
        # df_clust_all = pd.DataFrame()
        segdaid_all_lst = []
        now_all = datetime.now()
        df_sc_all = df_sc.copy()
        
        for i in df_sc_all['SEG_NO']:
            # string = request['segdaid']+'_'+'{}'.format(i)
            string = int(request['segdaid']+'{}'.format(i))
            segdaid_all_lst.append(string)
        segdaid_df = pd.DataFrame(segdaid_all_lst)
        df_sc_all['SEG_ID'] = segdaid_df

        if request['iutype'] == 'IR':
            # df_clust_all['ITEM_CD'] = df_sc['ITEM_CD']
            df_sc_all['SEG_TYPE'] = 'ML'
            df_sc_all['LOAD_DT'] = now_all
            df_clust_all = df_sc_all[['ITEM_CD','SEG_TYPE','SEG_ID','LOAD_DT']]
        else:
            # df_clust_all['USER_CD'] = df_sc['USER_CD'].astype(int)
            # df_clust_all['USER_CD'] = data_id
            
            df_sc_all['SEG_TYPE'] = 'ML'
            df_sc_all['UPD_DT'] = now_all
            df_clust_all = df_sc_all[['USER_CD','SEG_TYPE','SEG_ID','UPD_DT']]
        
            # print(df_clust_all.shape)
            # print('-------------------------')
            # print(df_clust_all.head(10))

        # Nan 값 처리 
        df_clust_all = df_clust_all.dropna(axis=0)
        len_data = df_sc.shape[0]


        #10000개 샘플링 -> 1000개로 변경
        if len_data > 1000:
            split = StratifiedShuffleSplit(n_splits=1, test_size=1000/len_data, random_state=42)
            for train_index, test_index in split.split(df_sc, df_sc['SEG_NO']):
                
                # 2022.08.14 loc ---> iloc으로 변경
                strat_train_set = df_sc.iloc[train_index]
                strat_test_set = df_sc.iloc[test_index]
            data = strat_test_set
            if request['iutype'] == 'IR':
                data_id = data['ITEM_CD']
                data_id = data_id.reset_index(drop=True)
                data = data[data.columns.difference(['ITEM_CD'])]
                data = data.reset_index(drop=True)
            else:
                data_id = data['USER_CD']
                data_id = data_id.astype('str')
                data_id = data_id.reset_index(drop=True)
                data = data[data.columns.difference(['USER_CD'])]  
                data = data.reset_index(drop=True)
        else:
            if request['iutype'] == 'IR':
                data = df_sc[df_sc.columns.difference(['ITEM_CD'])]
                data_id = df_sc['ITEM_CD']
            else:
                data = df_sc[df_sc.columns.difference(['USER_CD'])]
                data_id = df_sc['USER_CD']

        # t-SNE 

        tSNE=TSNE(n_components=2)
        tSNE_result=tSNE.fit_transform(data)

        x=tSNE_result[:,0]
        y=tSNE_result[:,1]
        y_kmeans = np.array(data['SEG_NO'])


        now = datetime.now()

        clust_result = pd.DataFrame()
        clust_result_for_db = pd.DataFrame()

        #DB insert -----------------------------------
        clust_result_for_db['seg_no'] = y_kmeans

        segdaid_lst = []
        for i in y_kmeans:
            string = int(request['segdaid']+'{}'.format(i))
            segdaid_lst.append(string)
        segdaid_df = pd.DataFrame(segdaid_lst)
        clust_result_for_db['SEG_ID'] = segdaid_df

        if request['iutype'] == 'IR':
            clust_result_for_db['ITEM_CD'] = data_id
            clust_result_for_db['SEG_TYPE'] = 'ML'
            clust_result_for_db['LOAD_DT'] = now
            clust_result_for_db = clust_result_for_db[['ITEM_CD','SEG_TYPE','SEG_ID','LOAD_DT']]
        else:
            clust_result_for_db['USER_CD'] = data_id
            clust_result_for_db['SEG_TYPE'] = 'ML'
            clust_result_for_db['UPD_DT'] = now
            clust_result_for_db = clust_result_for_db[['USER_CD','SEG_TYPE','SEG_ID','UPD_DT']]
    
    
        sql_data = list(map(tuple,df_clust_all.values))
        cursor=conn.cursor()
        cursor.execute(query)

        conn.commit()
        
        # INSERT
        if request['iutype'] == 'IR':
            cursor.execute('TRUNCATE TABLE 테이블')
            conn.commit()
            cursor.executemany('INSERT INTO 테이블(ITEM_CD,TYPE,S_ID,LOAD_DT) VALUES (%s,%s,%s,%s)',sql_data)
            conn.commit()
        else:
            cursor.execute('TRUNCATE TABLE 테이블2')
            conn.commit() 
            cursor.executemany('INSERT INTO 테이블2(USER_CD,TYPE,S_ID,LOAD_DT) VALUES (%s,%s,%s,%s)',sql_data)
            conn.commit()

        #----------------------------------------------

        clust_result['x1'] = x
        clust_result['x2'] = y
        clust_result['seg_no'] = y_kmeans

        segres_data = OrderedDict()
        segres_data['version'] = "1.0"
        segres_data['segda_id'] = request['segdaid'] 
        segres_data['seg_cnt'] = optimize_k  
        segres_data['data'] = clust_result.to_dict('records')
    
    return segres_data

# DAG 생성
with DAG(
   dag_id = 'pipeline',
#    default_args = default_args,
   description = "",
   schedule_interval = None,
   start_date = airflow.utils.dates.days_ago(10),
   ) as dag:

    # 시작 Task
    task_0 = DummyOperator(task_id='init')

    seg_extract_result = seg_extract()

    task_0 >> seg_extract_result

