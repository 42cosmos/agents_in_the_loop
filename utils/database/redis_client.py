import redis
from redis.commands.search.field import VectorField, TagField, NumericField, TextField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType


class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_conn = redis.Redis(host=host, port=port, db=db)

    def get_all_keys(self, pattern='*'):
        return [key.decode('utf-8') for key in self.redis_conn.keys(pattern)]

    def get_values_by_key_pattern(self, pattern):
        keys = self.get_all_keys(pattern)
        return {key: self.redis_conn.get(key).decode('utf-8') if self.redis_conn.get(key) else None for key in keys}

    def set_value(self, key, value):
        self.redis_conn.set(key, value)

    def get_value(self, key):
        value = self.redis_conn.get(key)
        if value:
            return value.decode('utf-8')
        return None

    def add_to_set(self, key, *values):
        self.redis_conn.sadd(key, *values)

    def get_all_from_set(self, key):
        return [value.decode('utf-8') for value in self.redis_conn.smembers(key)]

    def push_to_list(self, key, *values):
        self.redis_conn.lpush(key, *values)

    def get_all_from_list(self, key):
        return [value.decode('utf-8') for value in self.redis_conn.lrange(key, 0, -1)]

    def delete_key(self, key):
        return self.redis_conn.delete(key)


class RedisVector(RedisClient):
    def __init__(self,
                 dataset_title_value: str,
                 dataset_lang_value: str,
                 host: str = 'localhost',
                 port: int = 6379,
                 db=0,
                 index_name="dataset_embeddings",
                 doc_prefix="dataset:",
                 dataset_field_name: str = "dataset_name",
                 dataset_lang_field_name: str = "dataset_language",
                 model_field_name: str = "model",
                 vector_field_name: str = "embedding",
                 embedding_size=768,
                 remove_history=False
                 ):
        super().__init__(host, port, db)

        self.dataset_field_name = dataset_field_name
        self.dataset_title_value = dataset_title_value

        self.dataset_lang_field_name = dataset_lang_field_name
        self.dataset_lang_value = dataset_lang_value

        self.model_field_name = model_field_name
        self.vector_field_name = vector_field_name

        self.embedding_size = embedding_size

        self.index_name = index_name
        self.doc_prefix = doc_prefix

        if remove_history:
            self.delete_data()
        self._set_schema()

    def _get_id_key(self, model_name, data_id):
        return f"{self.doc_prefix}{self.dataset_title_value}:{self.dataset_lang_value}:{model_name}:{data_id}"

    def insert_vectors(self, model_name: str, ids, vectors):
        assert len(ids) == len(vectors), "Names and vectors must have the same length !"

        pipeline = self.redis_conn.pipeline()

        for id_, vector in zip(ids, vectors):
            self._insert_single_vector(model_name=model_name, data_id=id_,
                                       vector=vector, pipeline=pipeline)
        pipeline.execute()

    def _insert_single_vector(self, model_name, data_id, vector, pipeline=None):
        document = {
            self.dataset_field_name: self.dataset_title_value,
            self.dataset_lang_field_name: self.dataset_lang_value,
            self.model_field_name: model_name,
            self.vector_field_name: vector.tobytes()
        }
        id_key = self._get_id_key(model_name, data_id)

        if pipeline:
            pipeline.hset(id_key, mapping=document)
            pipeline.ft(self.index_name).add_document(id_key, **document, replace=True)
        else:
            self.redis_conn.hset(id_key, mapping=document)
            self.redis_conn.ft(self.index_name).add_document(id_key, **document, replace=True)

    def delete_data(self):
        """
        BEWARE ! Delete all data in redis
        :return:
        """
        self.redis_conn.flushall()

    def _set_schema(self, algorithm="HNSW", distance_metric="L2"):
        try:
            self.redis_conn.ft(self.index_name).info()
            print("Index already exists ! ")

        except:
            schema = (
                TagField(self.dataset_field_name),
                TagField(self.dataset_lang_field_name),
                TagField(self.model_field_name),
                VectorField(self.vector_field_name,
                            algorithm,
                            {"TYPE": "FLOAT32",
                             "DIM": self.embedding_size,
                             "DISTANCE_METRIC": distance_metric}),
            )
            definition = IndexDefinition(prefix=[self.doc_prefix], index_type=IndexType.HASH)

            self.redis_conn.ft(self.index_name).create_index(fields=schema, definition=definition)

    def get_vector(self, name, key):
        """
        :param name: 데이터베이스 key 값
        :param key: 스키마의 key 값
        """
        return self.redis_conn.hget(name, key)

    def get_similar_vector_id(self, model_name, vector, num=10):
        similarity_query = f'(@{self.dataset_field_name}:{{{{{self.dataset_title_value}}}}} ' \
                           f'@{self.dataset_lang_field_name}:{{{{{self.dataset_lang_value}}}}} ' \
                           f'@{self.model_field_name}:{{{{{model_name}}}}})' \
                           f'=>[KNN {num} @{self.vector_field_name} $vec_param AS dist]'
        q = Query(similarity_query).sort_by('dist')
        vector_params = {"vec_param": vector.tobytes()}
        res = self.redis_conn.ft().search(q, query_params=vector_params)
        # 데이터셋 아이디만 추출
        doc_ids = [doc.id for doc in res.docs]
        return doc_ids
