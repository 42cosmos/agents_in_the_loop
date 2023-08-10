import redis
from redis.commands.search.field import VectorField, TagField, NumericField, TextField
from redis.commands.search.query import Query


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
                 model_title_value: str,
                 host: str='localhost',
                 port: int=6379,
                 db=0,
                 dataset_field_name: str="dataset",
                 dataset_lang_field_name:str="language",
                 model_field_name: str="model",
                 vector_field_name: str="embedding",
                 embedding_size=256,
                 ):
        super().__init__(host, port, db)

        self.dataset_title_value = dataset_title_value
        self.dataset_lang_value = dataset_lang_value
        self.model_title_value = model_title_value
        self.dataset_field_name = dataset_field_name
        self.dataset_lang_field_name = dataset_lang_field_name
        self.model_field_name = model_field_name
        self.vector_field_name = vector_field_name
        self.embedding_size = embedding_size

    def set_vector(self, name, *values):

        self.redis_conn.hset(
            name,
            mapping={self.dataset_field_name: self.dataset_title_value,
                     self.dataset_lang_field_name: self.dataset_lang_value,
                     self.model_field_name: self.model_title_value,
                     self.vector_field_name: values},
        )

    def delete_data(self):
        self.redis_conn.flushall()

    def _set_schema(self, algorithm="HNSW"):
        schema = (
            TagField(self.dataset_field_name),
            TagField(self.dataset_lang_field_name),
            TagField(self.model_field_name),
            VectorField(self.vector_field_name,
                        algorithm,
                        {"TYPE": "FLOAT64", "DIM": self.embedding_size, "DISTANCE_METRIC": "L2"}),
        )

        self.redis_conn.ft().create_index(schema)

    def get_vector(self, name):
        return self.redis_conn.hget(name, self.vector_field_name)

    def get_similar_vectors(self, name, num=10):
        # 주어진 name에서 벡터 및 필요한 메타데이터 정보를 가져옵니다.
        vector = self.redis_conn.hget(name, self.vector_field_name)
        dataset = self.redis_conn.hget(name, self.dataset_field_name).decode('utf-8')
        language = self.redis_conn.hget(name, self.dataset_lang_field_name).decode('utf-8')
        model = self.redis_conn.hget(name, self.model_field_name).decode('utf-8')

        # 쿼리를 구성합니다.
        q = Query(
            f'(@{self.dataset_field_name}:{dataset} @{self.dataset_lang_field_name}:{language} @{self.model_field_name}:{model})=>[KNN {num} @{self.vector_field_name} {vector} AS dist]').sort_by(
            'dist')

        # 검색을 수행하고 결과를 반환합니다.
        res = self.redis_conn.ft().search(q)

        return res
