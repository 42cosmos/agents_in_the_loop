import json
import logging
from typing import Any, List, Dict

import redis
from redis.commands.search.field import (
    VectorField,
    TagField,
    NumericField,
    TextField
)
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType


class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_conn = redis.Redis(host=host, port=port, db=db)

    def get_information_by_index_name(self, index_name):
        try:
            return self.redis_conn.ft(index_name).info()
        except Exception as e:
            logging.error(f"Error in get_information_by_index_name: {e}")
            raise

    def get_all_key_by_pattern(self, pattern='*'):
        try:
            return [key.decode('utf-8') for key in self.redis_conn.keys(pattern)]
        except Exception as e:
            logging.error(f"Error in get_all_key_by_pattern: {e}")
            raise

    def get_values_by_key_pattern(self, pattern):
        keys = self.get_all_key_by_pattern(pattern)
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
        try:
            return self.redis_conn.delete(key)
        except Exception as e:
            logging.error(f"Error in delete_key: {e}")
            raise

    def delete_documents_by_index_name(self, index_name):
        """
        BEWARE ! Delete all data in redis
        """
        try:
            results = self.redis_conn.ft(index_name).search(Query('*'))
            for doc in results.docs:
                self.redis_conn.delete(doc.id)
            logging.info(f"Deleted {len(results.docs)} documents from {index_name}")
        except Exception as e:
            logging.error(f"Error in delete_documents_by_index_name: {e}")
            raise


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
            self.delete_documents_by_index_name(self.index_name)
        self.set_schema()

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

    def set_schema(self, algorithm="HNSW", distance_metric="L2"):
        try:
            self.redis_conn.ft(self.index_name).info()
            logging.info(f"{self.index_name} Index already exists ! ")

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
            logging.info(f"Index {self.index_name} created !")

    def get_vector(self, data_id, field_name):
        """
        :param data_id: 데이터베이스 key 값
        :param field_name: 스키마의 key 값
        """
        try:
            return self.redis_conn.hget(data_id, field_name)
        except Exception as e:
            logging.error(f"Error in get_vector: {e}")
            raise

    def get_similar_vector_id(self, model_name, vector, num=10):
        similarity_query = f'(@{self.dataset_field_name}:{{{{{self.dataset_title_value}}}}} ' \
                           f'@{self.dataset_lang_field_name}:{{{{{self.dataset_lang_value}}}}} ' \
                           f'@{self.model_field_name}:{{{{{model_name}}}}})' \
                           f'=>[KNN {num} @{self.vector_field_name} $vec_param AS dist]'
        q = Query(similarity_query).sort_by('dist')
        vector_params = {"vec_param": vector.tobytes()}
        try:
            res = self.redis_conn.ft().search(q, query_params=vector_params)
        except Exception as e:
            logging.error(f"Error in get_similar_vector_id: {e}")
            raise
        # 데이터셋 아이디만 추출
        doc_ids = [doc.id for doc in res.docs]
        return doc_ids


class RedisPrompt(RedisClient):
    def __init__(self,
                 host: str = 'localhost',
                 port: int = 6379,
                 db=0,
                 index_name="conversation",
                 prompt_field_name="prompt",
                 doc_prefixes=None,
                 remove_history=False):

        super().__init__(host, port, db)

        if doc_prefixes is None:
            doc_prefixes = ["memory", "prompt"]

        self.index_name = index_name
        self.prompt_field_name = prompt_field_name
        self.doc_prefixes = doc_prefixes

        if remove_history:
            self.delete_documents_by_index_name(self.index_name)
        self.set_prompt_schema()

    def get_prompt_id(self, id_, doc_prefix="memory"):
        doc_prefix = doc_prefix.lower()
        assert doc_prefix in self.doc_prefixes, "doc_prefix must be memory or prompt!"
        set_doc_prefix = doc_prefix if doc_prefix.endswith(":") else doc_prefix + ":"

        return f"{set_doc_prefix}{id_}"

    def set_prompts(self, prompt_datum: List[Dict], doc_prefix="memory"):
        """
        :param doc_prefix: memory or prompt
        :param prompt_datum: dictionaries in List with keys prompt, initial_answer, feedback, final_answer
        """
        try:
            pipeline = self.redis_conn.pipeline()

            for dict_data in prompt_datum:
                # TODO: set_prompt로 들어오는 datum 재정의
                data_id = dict_data["id"] # memory:student:{data_id}:{#}
                data_id = self.get_prompt_id(data_id, doc_prefix)
                prompt_data = dict_data["llm_answer"]

                self._set_single_prompt(data_id=data_id,
                                        prompt_data=prompt_data,
                                        doc_prefix=doc_prefix,
                                        pipeline=pipeline)


        except Exception as e:
            db_info = self.get_information_by_index_name(self.index_name)
            num_docs, num_failure = db_info["num_docs"], db_info["hash_indexing_failures"]
            logging.error(f"Error in set_prompt : {e}, num_docs : {num_docs}, num_failure : {num_failure}")
            raise

    def _set_single_prompt(self, data_id, prompt_data: dict, doc_prefix="memory", pipeline=None):
        prompt_key = self.get_prompt_id(data_id, doc_prefix)
        serialised_data = json.dumps(prompt_data)
        if pipeline:
            pipeline.set(prompt_key, serialised_data)
            pipeline.ft(self.index_name).add_document(prompt_key, **prompt_data, replace=True)
        else:
            self.redis_conn.set(prompt_key, serialised_data)
            self.redis_conn.ft(self.index_name).add_document(prompt_key, **prompt_data, replace=True)
            logging.info(f"Prompt {prompt_key} added !")

    def set_prompt_schema(self):
        try:
            self.redis_conn.ft(self.index_name).info()
            logging.info(f"{self.index_name} Index already exists ! ")

        except Exception as e:
            schema = (
                TextField(self.prompt_field_name),
            )
            definition = IndexDefinition(prefix=[self.doc_prefixes], index_type=IndexType.HASH)

            self.redis_conn.ft(self.index_name).create_index(fields=schema, definition=definition)
            logging.info(f"Index {self.index_name} created !")

    def get_prompt(self, data_id, doc_prefix="memory"):
        data_id_for_search = self.get_prompt_id(data_id, doc_prefix)
        try:
            prompt_value = self.redis_conn.hget(data_id_for_search, self.prompt_field_name)
            if prompt_value:
                return prompt_value.decode("utf-8")
            else:
                logging.info(f"Prompt {data_id} not found !")

        except Exception as e:
            logging.error(f"Error in get_prompt : {e}")
            raise

    def get_prompt_json(self, data_id, doc_prefix="memory"):
        data_id_for_search = self.get_prompt_id(data_id, doc_prefix)
        try:
            serialised_data = self.redis_conn.get(data_id_for_search)
            if serialised_data:
                return json.loads(serialised_data)  # 문자열을 다시 JSON 객체로 변환
            else:
                logging.info(f"Prompt {data_id} not found !")
        except Exception as e:
            logging.error(f"Error in get_prompt_json : {e}")
            raise
