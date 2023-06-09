# library
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, desc , col, max
from pyspark.ml.feature import  StringIndexer
from pyspark.sql import Row
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as f
from pyspark.sql import Window
import datetime 
from datetime import timedelta 

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, desc , col, max
from pyspark.ml.feature import  StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.feature import  StringIndexer
from pyspark.ml import Pipeline

from pyspark.sql.functions import col
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import when
from pyspark.sql.functions import lit
import os 

spark = SparkSession.builder \
                            .master("yarn") \
                            .appName("PySpark") \
                            .config("spark.executor.memory", "12g") \
                            .config("spark.executor.cores", 4) \
                            .config("spark.executor.instances", 10) \
                            .config("spark.driver.memory","30g") \
                            .getOrCreate()
                            

# Function
def createRating(A_df,B_df,C_df,user_clust_mstr,clust_id):
    
    """
    1) ALS 추천 모델을 INPUT 데이터
    1. 클러스터 번호별로 회원 데이터 불러오기
    2. A 테이블이랑 조인해서, 제외 A 반영
    3. A 데이터 불러오기 
    4. (선호도) --> 선호도 (조건에 따른 A수)
    5. (선호도 점수 계산) --> 조건 선호도에 따라 선호도 점수 부여* 
    """
    
    # sparkSQL temp table
    A_df.createOrReplaceTempView("A_df")
    B_df.createOrReplaceTempView("B_df")
    C_df.createOrReplaceTempView("C_df")
    user_clust_mstr.createOrReplaceTempView("user_clust_mstr")
    
    
    # sparkSQL 
    sql1 = """
        SELECT 유저, 아이템, 기간
        FROM A_df A
        INNER JOIN user_clust_mstr C ON A.유저 = C.유저
        AND C.ID = """+str(clust_id)+\
        """ GROUP BY 유저, 아이템, 기간
        """
    
    sql2 = """
            SELECT 아이템1,아이템2
            FROM C_df A 
            INNER JOIN B_df B ON A.아이템2 = B.아이템2
            """
    
    sql3 = """
        SELECT 아이템2,아이템3
        FROM B_df
       """
    
    sql4 = """
            select 유저,아이템3,COUNT(distinct 아이템2) AS 아이템2_cnt
            from A_df A
            inner join C_df B ON A.아이템1 = B.아이템1
            inner join B_df C ON B.아이템2 = C.아이템2
            group by 유저,아이템3
           """
    
    A_df2 = spark.sql(sql1)
    
    A_df2.createOrReplaceTempView("A_df")
    
    C_df = spark.sql(sql2)
    C_df.createOrReplaceTempView("C_df")
    
    B_df = spark.sql(sql3)
    B_df.createOrReplaceTempView("B_df")
    
    utc = spark.sql(sql4)
    utc.createOrReplaceTempView("utc")
    
    # rating
    sql5 = """
            select a3.유저,a3.아이템1,(case when a4.아이템2_cnt is null then 1
                                                when a4.아이템2_cnt = 2 then 2
                                                when a4.아이템2_cnt IN (3,4) THEN 3
                                                when a4.아이템2_cnt IN (5,6) THEN 4
                                                when a4.아이템2_cnt > 6 THEN 5
                                                else 1 end) AS rating
            from (
                    select a1.유저,a1.아이템1,a2.아이템3
                    from (
                            select A.유저,A.아이템1,B.아이템2
                            from A_df A
                            left join C_df B ON A.아이템1 = B.아이템1
                        ) a1
                    left join B_df a2 ON a1.아이템2 = a2.아이템2
                    group by a1.유저,a1.아이템1,a2.아이템3
                )a3
            left join utc a4 on a3.유저 = a4.유저 and a3.아이템3 = a4.아이템3

            """
    
    rating = spark.sql(sql5).cache()

    rating = rating.na.drop()
    
    # integer 변환
    rating = rating.withColumn("유저",rating.유저.cast('int')).withColumn("아이템1",rating.아이템1.cast('int'))
    rating_df = rating.select('유저','아이템1','rating').orderBy('유저')
  
    C_df.unpersist()
    B_df.unpersist()
    utc.unpersist()
    return rating_df


def recent_item_extractor(A_df,user_clust_mstr,clust_id):
    
    """
    2) 아이템 기반 추천 기준 (최근 * )
    1. 클러스터별 유저별 로그 데이터
    2. 최근 특정 조건 반영
    """
    A_df.createOrReplaceTempView("A_df")
    user_clust_mstr.createOrReplaceTempView("user_clust_mstr")
    
    A_df_tmp = A_df_create(A_df,user_clust_mstr,clust_id)

    # 조건값이 70% 이상의 아이템중 최근기준 (70%의 이상 없으면, 최근 아이템)
    recent_curr_log_raw = A_df_tmp.withColumn("기간(년도)",when(A_df_tmp.s_rate >= 70.00,"Y").otherwise("N"))
    
    w1 = Window.partitionBy('유저').orderBy(col("기간").desc(),col("기간(년도)").desc())
    
    recent_curr_log1 = recent_curr_log_raw.withColumn('row_number',f.row_number().over(w1))

    recent_curr_log2 = recent_curr_log1.filter(recent_curr_log1.row_number == 1)
    
    recent_curr_log1.unpersist()
    recent_curr_log_raw.unpersist()
    
    return recent_curr_log2

def A_df_create(A_df,user_clust_mstr,clust_id):
    
    # 클러스터_유저 필터링
    sql = """
         select A.* from A_df A 
         INNER JOIN user_clust_mstr B ON A.유저 = B.유저
         AND B.S_DA_ID = """+str(clust_id)
    
    A_df_tmp = spark.sql(sql).cache()
    return A_df_tmp

def train_als_model(rating_df):
    
    """
    3) ALS 추천 모델 학습 
    
    1. 추천 모델 학습 (ALS)
    2. TRAIN/TEST SPLIT (운영 코드에서는 제외)
    3. 추천 모델과 아이템 벡터 추출
    
    """
    
    USERID = '유저'
    ITEMID = '아이템1'
    COUNT = 'rating'

    als = ALS(maxIter=5,regParam=0.01,userCol=USERID,itemCol=ITEMID,ratingCol=COUNT)
    model = als.fit(rating_df)
    itemFactors = model.itemFactors
    
    return(model,itemFactors)

def user_item_matrix(training,model,itemFactors):
    
    """
    4) 사용자-아이템 매트릭스 (CROSS-JOIN) 
    
    1. USER-ITEM MATRIX 생성
    2. ItemFactors 추출
    3. 전체 user-item 예측 결과 (아이템/사용자 유사도 모두 포함한 예측 결과)
    
    """
        
    # Get the cross join of all user-item pairs and score them.
    users = training.select('유저').distinct()
    items = training.select('아이템1').distinct()
    user_item = users.crossJoin(items)
    dfs_pred = model.transform(user_item)
    itemFactors = model.itemFactors

    # Remove seen items.
    dfs_pred_exclude_train = dfs_pred.alias("pred").join(
        training.alias("train"),
        (dfs_pred['유저'] == training['유저']) & (dfs_pred['아이템1'] == training['아이템1']),
        how='outer'
    )

    dfs_pred_final = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train.Rating"].isNull()) \
        .select('pred.' + '유저', 'pred.' + '아이템1', 'pred.' + "prediction")
    
    dfs_pred_exclude_train.unpersist()
    dfs_pred.unpersist()
    users.unpersist()
    items.unpersist()
    
    return dfs_pred_final


def item_item_similarity(itemFactors):
    
    """
    5) 아이템 유사도 추출 
    
    1. 아이템 벡터에서 코사인 유사도 계산
    2. 아이템 유사도 매트릭스 추출
    
    """
        
    ## Item-Item feature DataFrame
    size = itemFactors.limit(1).select(F.size('features')).first()[0]
    df = itemFactors.crossJoin(itemFactors.select(F.col("id").alias("id_b"), F.col("features").alias("features_item")))

    # item-item cosine similarity
    similar_item_df = df.select(
    'id',
    'id_b',
    sum([F.col('features')[i] * F.col('features_item')[i] for i in range(size)]).alias('dot_product'),
    F.sqrt(sum([F.col('features')[i] * F.col('features')[i] for i in range(size)])).alias('norm1'),
    F.sqrt(sum([F.col('features_item')[i] * F.col('features_item')[i] for i in range(size)])).alias('norm2')
    ).selectExpr(
        'id',
        'id_b',
        'dot_product / norm1 / norm2 as cosine_similarity'
    )
    
    # 0보다 큰 경우
    similar_item_df_tmp = similar_item_df.filter(similar_item_df.cosine_similarity > 0)
    similar_item_df = similar_item_df_tmp.filter(similar_item_df.id != similar_item_df.id_b)
    
    similar_item_df_tmp.unpersist()
    df.unpersist()
    
    return similar_item_df

def get_rc_f_rc_c(similar_item_df,recent_curr_log2,C_df,B_df,GR):
    
    recent_curr_log2 = recent_curr_log2.withColumn("아이템1",recent_curr_log2.아이템1.cast('int'))
    C_df_trg = grade_curri(GR,C_df,B_df).withColumnRenamed('아이템1', '아이템1').withColumnRenamed('아이템2', '아이템2')
    
    similar_item_df.createOrReplaceTempView("similar_item_df")
    recent_curr_log2.createOrReplaceTempView("recent_curr_log2")
    C_df_trg.createOrReplaceTempView("C_df_trg")
   
    sql = """
            SELECT B.유저
                  ,B.아이템1 AS RECENT_아이템1
                  ,A.id_b AS RECO_아이템1
                  ,A.cosine_similarity AS RECO_VALUE
                  ,ROW_NUMBER() OVER(PARTITION BY B.유저 ORDER BY A.cosine_similarity DESC) AS C_order
                  ,C.아이템2
                  ,C.GR
            FROM similar_item_df A 
            INNER JOIN recent_curr_log2 B ON A.id = B.아이템1
            INNER JOIN C_df_trg C ON A.id_b = C.아이템1
        """
        
    similar_item_df.unpersist()
    recent_curr_log2.unpersist()

    f_r_c = spark.sql(sql)
    return f_r_c,C_df_trg

def grade_curri(GR,C_df,B_df):
    
    C_df.createOrReplaceTempView("C_df")
    B_df.createOrReplaceTempView("B_df")
    
    if GR == '1':
        sql2_1 = """
            SELECT A.아이템1 AS 아이템1,A.아이템2 AS 아이템2,아이템4,1 AS GR
            FROM C_df A 
            INNER JOIN B_df B ON A.아이템2 = B.아이템2
            AND B.G_CD IN ('0002','0006','0010','0014')
            """
        C_df = spark.sql(sql2_1)   
    elif GR == '2':
        sql2_2 = """
            SELECT A.아이템1 AS 아이템1,A.아이템2 AS 아이템2,아이템4,2 AS GR
            FROM C_df A 
            INNER JOIN B_df B ON A.아이템2 = B.아이템2
            AND B.G_CD IN ('대외비')
            """
        C_df = spark.sql(sql2_2)
    else:
        sql2_3 = """
            SELECT A.아이템1 AS 아이템1,A.아이템2 AS 아이템2,아이템4,3 AS GR
            FROM C_df A 
            INNER JOIN B_df B ON A.아이템2 = B.아이템2
            AND B.G_CD IN ('대외비')
            """
        C_df = spark.sql(sql2_3)
    return C_df

def upload_result_to_hdfs(f_rc_cr_rslt,clust_id):
    
    now = datetime.datetime.now()+timedelta(days=1)
    current_date = now.strftime('%Y%m%d')
        
    path = 's3://bucket/output/'+str(current_date)+'/user1/'
    output_source = path+str(clust_id)+"_als_result.csv"
    f_rc_cr_rslt.repartition(100).write.mode("overwrite").option("mapreduce.fileoutputcommiitter.algorithm.version","2").option('header','true').option("delimiter","\t").csv(output_source)
    f_rc_cr_rslt.unpersist()
    return (print('UPLOAD Success!'))    
    


# 01. start_log 
now = datetime.datetime.now()+timedelta(days=1)
current_date = now.strftime('%Y%m%d')
start_log = spark.createDataFrame([(current_date,'Started')],['date','start_massage'])
output_source = "s3n://start/"+str(current_date)+'_u1_start_log.csv'
start_log.coalesce(1).write.mode('overwrite').option('header','true').csv(output_source)

# 02.load_data 

path = "s3://bucket/input/" +str(current_date)
data_source1 = path+"/FCS_"+str(current_date)+".csv"
data_source2 = path+"/RL_"+str(current_date)+".csv"
data_source3 = path+"/DC_"+str(current_date)+".csv"
data_source4 = path+"/FLS_"+str(current_date)+".csv"
data_source6 = "s3://bucket/output/segment/"+str(current_date)+'/user_seg_result_'+ str(current_date) +'.csv'
data_source7 = path+"/DM_"+str(current_date)+".csv"

A_df = spark.read.options(header="True").option("encoding","utf-8").csv(data_source1).cache()
B_df = spark.read.options(header="True").option("encoding","utf-8").csv(data_source2).cache()
C_df = spark.read.options(header="True").option("encoding","utf-8").csv(data_source3).cache()
l_A_df = spark.read.options(header="True").option("encoding","utf-8").csv(data_source4).cache()
user_df = spark.read.options(header="True").option("encoding","utf-8").csv(data_source7).cache()
user_clust_mstr = spark.read.options(header="True").option("encoding","utf-8").csv(data_source6).cache()

# sparkSQL temp table
A_df.createOrReplaceTempView("A_df")
B_df.createOrReplaceTempView("B_df")
C_df.createOrReplaceTempView("C_df")
user_clust_mstr.createOrReplaceTempView("user_clust_mstr")

sql = """SELECT S_DA_ID,COUNT(DISTINCT 유저) AS user_cnt 
         FROM user_clust_mstr
         GROUP BY S_DA_ID """
user_clust_cnt = spark.sql(sql).cache()

user_cnt_max_value = user_clust_cnt.agg({"user_cnt": "max"}).collect()[0]
user_cnt_max_value = user_cnt_max_value["max(user_cnt)"]

big_cluster = user_clust_cnt.filter(user_clust_cnt.user_cnt >= user_cnt_max_value).select('SEGDA_ID').collect()

for cluster in big_cluster:
    cluster_id = str(cluster['S_DA_ID'])
    user_clust_mstr = user_clust_mstr.filter(user_clust_mstr.S_DA_ID != str(cluster_id))
    user_clust_mstr2 = user_clust_mstr.filter(user_clust_mstr.S_DA_ID < str(2000))

user_clust_mstr3 = user_clust_mstr2.join(user_df.select('유저','GR'),['유저'])
user_clust_df =user_clust_mstr3.select('GR','S_DA_ID').distinct().collect()

for user_clust_row in user_clust_df:
    clust_id = str(user_clust_row['S_DA_ID'])
    GR = str(user_clust_row['GR'])
    rating_df = createRating(A_df,B_df,C_df,user_clust_mstr,clust_id).cache()
    recent_curr_log2 = recent_item_extractor(A_df,user_clust_mstr,clust_id).cache()
    model,itemFactors= train_als_model(rating_df)
    dfs_pred_final = user_item_matrix(rating_df,model,itemFactors).cache()
    similar_item_df = item_item_similarity(itemFactors).cache()
    f_r_c,C_df_trg = get_rc_f_rc_c(similar_item_df,recent_curr_log2,C_df,B_df,GR)
    
    l_A_df = l_A_df.withColumnRenamed('유저', '유저_A').withColumnRenamed('아이템2', '아이템2_B').cache()
    
    # 제외 처리
    m_l_exclu = f_r_c.join(
    l_A_df.select(l_A_df.유저_A,l_A_df.아이템2_B),(f_r_c.유저 == l_A_df.유저_A) & (f_r_c.아이템2  == l_A_df.아이템2_B),
    how= 'outer')
    f_c_l_exclu = m_l_exclu.filter(m_l_exclu.아이템2_B.isNull()).cache()
    
    # 150
    f_c_l_exclu.createOrReplaceTempView("f_c_l_exclu")
    
    sql = """
        SELECT * 
        FROM (
                SELECT 유저,RECENT_아이템1,RECO_아이템1,RECO_VALUE,GR,ROW_NUMBER() OVER(PARTITION BY 유저 ORDER BY RECO_VALUE DESC) AS C_order
                FROM f_c_l_exclu
            ) AA 
        WHERE C_order <= 150
        """

    f_rc_cr_rslt = spark.sql(sql).cache()

    f_c_l_exclu.unpersist()
    l_A_df.unpersist()
    f_r_c.unpersist()
    C_df_trg.unpersist()
    similar_item_df.unpersist()
    recent_curr_log2.unpersist()
    rating_df.unpersist()

    A_df.unpersist() 
    B_df.unpersist()
    C_df.unpersist()
    l_A_df.unpersist()
    user_df.unpersist()
    user_clust_mstr.unpersist() 

        
    upload_result_to_hdfs(f_rc_cr_rslt,clust_id)

now = datetime.datetime.now()+timedelta(days=1)
finish_time = now.strftime('%m%d%Y, %H:%M:%S')

finish_log = spark.createDataFrame(
[(current_date,'Finished')],[finish_time,'finish_massage test!'])

output_source = "s3n://log/"+str(current_date)+'_finish_u1_log.csv'
finish_log.coalesce(1).write.mode('overwrite').option('header','true').csv(output_source)   

spark.stop()
