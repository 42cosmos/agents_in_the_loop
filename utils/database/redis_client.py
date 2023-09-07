import json
import logging
import os
from typing import List, Dict

import redis
from colorama import Fore
from redis.commands.search.field import (
    VectorField,
    TagField,
    NumericField,
    TextField
)
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from dotenv import load_dotenv

from utils.llm.base import ChatModelResponse


class RedisClient:
    def __init__(self, env_path="/home/eunbinpark/workspace/agents_in_the_loop/utils/database/redis.env"):
        load_dotenv(dotenv_path=env_path)
        host = os.getenv("REDIS_HOST", None)
        assert host, "REDIS_HOST is not set !"

        port = os.getenv("REDIS_PORT", None)
        assert port, "REDIS_PORT is not set !"

        db = os.getenv("REDIS_DB", None)
        assert db, "REDIS_DB is not set !"

        password = os.getenv("REDIS_PASSWORD", None)
        assert password, "REDIS_PASSWORD is not set !"

        self.logger = logging.getLogger(__class__.__qualname__)
        self.redis_conn = redis.Redis(host=host, port=port, db=db, password=password)

    def get_all_key_by_pattern(self, pattern='*'):
        try:
            return [key.decode('utf-8') for key in self.redis_conn.keys(pattern)]
        except Exception as e:
            self.logger.error(f"Error in get_all_key_by_pattern: {e}")
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
            self.logger.error(f"Error in delete_key: {e}")
            raise

    def delete_documents_by_doc_prefix(self, doc_prefix=None):
        """
        BEWARE ! Delete all data in redis
        """
        if doc_prefix is None:
            raise ValueError("doc_prefix must be specified !")

        try:
            if isinstance(doc_prefix, str):
                results = self.get_all_key_by_pattern(f"{doc_prefix}*")
                for doc_id in results:
                    self.redis_conn.delete(doc_id)
                self.logger.info(f"Deleted {len(results)} documents from {doc_prefix}")
            if isinstance(doc_prefix, list):
                for prefix in doc_prefix:
                    results = self.get_all_key_by_pattern(f"{prefix}*")
                    for doc_id in results:
                        self.redis_conn.delete(doc_id)
                    self.logger.info(f"Deleted {len(results)} documents from {prefix}")

        except Exception as e:
            self.logger.error(f"Error in delete_documents_by_index_name: {e}")
            raise


class RedisVector(RedisClient):
    def __init__(self,
                 dataset_title_value: str,
                 dataset_lang_value: str,
                 index_name="dataset_embeddings",
                 doc_prefix="dataset:",
                 dataset_field_name: str = "dataset_name",
                 dataset_lang_field_name: str = "dataset_language",
                 model_field_name: str = "model",
                 vector_field_name: str = "embedding",
                 embedding_size=768,
                 remove_history=False
                 ):
        super().__init__()
        self.logger = logging.getLogger(__class__.__qualname__)
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
            self.delete_documents_by_doc_prefix(self.doc_prefix)
        self.set_schema()

    def get_information_by_index_name(self, index_name=None):
        if index_name is None:
            index_name = self.index_name

        try:
            return self.redis_conn.ft(index_name).info()
        except Exception as e:
            self.logger.error(f"Error in get_information_by_index_name: {e}")
            raise

    def _get_id_key(self, model_name, data_id):
        return f"{self.doc_prefix}{model_name}:{data_id}"

    def insert_vectors(self, model_name: str, ids, vectors):
        assert len(ids) == len(vectors), "Names and vectors must have the same length !"

        pipeline = self.redis_conn.pipeline()

        for id_, vector in zip(ids, vectors):
            self._insert_single_vector(model_name=model_name, data_id=id_,
                                       vector=vector, pipeline=pipeline)
        pipeline.execute()

        self.logger.info(f"Inserted {len(ids)} vectors !")

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
            self.logger.info(f"{Fore.GREEN}{self.index_name}{Fore.RESET} Index already exists ! ")

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
            self.redis_conn.ft(self.index_name).config_set("default_dialect", 2)
            self.logger.info(f"Index {self.index_name} created !")

    def get_vector(self, model_name, data_id, field_name="embedding"):
        """
        :param model_name: 임베딩 모델 이름
        :param data_id: 데이터베이스 key 값, 데이터 번호만 입력
        :param field_name: 스키마의 key 값
        """
        data_key_for_redis = self._get_id_key(model_name=model_name, data_id=data_id)
        try:
            return self.redis_conn.hget(data_key_for_redis, field_name)
        except Exception as e:
            self.logger.error(f"Error in get_vector: {e}")
            raise

    def search_similar_vector_by_data_id(self, model_name, vector, num=10):
        similarity_query = f'(@{self.dataset_field_name}:{{{self.dataset_title_value}}} ' \
                           f'@{self.dataset_lang_field_name}:{{{self.dataset_lang_value}}} ' \
                           f'@{self.model_field_name}:{{{model_name}}})' \
                           f'=>[KNN {num} @{self.vector_field_name} $vec_param AS dist]'
        q = Query(similarity_query).sort_by('dist').dialect(2)
        vector_params = {"vec_param": vector if isinstance(vector, bytes) else vector.tobytes()}

        try:
            res = self.redis_conn.ft(self.index_name).search(q, query_params=vector_params)
        except Exception as e:
            self.logger.error(f"Error in search_similar_vector_by_data_id: {e}")
            raise
        # 데이터셋 아이디만 추출
        doc_ids = [doc.id for doc in res.docs]
        return doc_ids


class RedisLLMResponse(RedisClient):
    def __init__(self,
                 index_name: str = "conversation_memory",
                 doc_prefix=None,
                 remove_history=False):
        self.logger = logging.getLogger(__class__.__qualname__)
        super().__init__()

        if doc_prefix is None:
            doc_prefix = "memory"

        self.index_name = index_name
        self.doc_prefix = doc_prefix

        if remove_history:
            self.delete_documents_by_doc_prefix(self.doc_prefix)

    def _get_memory_id(self, id_, doc_prefix="memory"):
        doc_prefix = doc_prefix.lower()

        if id_.split(":")[0] in self.doc_prefix:
            return id_
        set_doc_prefix = doc_prefix if doc_prefix.endswith(":") else doc_prefix + ":"
        id_ = id_[1:] if id_.startswith(":") else id_
        return f"{set_doc_prefix}{id_}"

    def set_memory(self, document: dict, data_id: str, doc_prefix="memory"):
        self._set_single_memory(data_id=data_id,
                                document=document,
                                doc_prefix=doc_prefix)

    def _set_single_memory(self, data_id, document, doc_prefix="memory", pipeline=None):
        prompt_key = self._get_memory_id(data_id, doc_prefix)

        if pipeline:
            pipeline.hset(prompt_key, mapping=document)
        else:
            self.redis_conn.hset(prompt_key, mapping=document)
            self.logger.info(f"Prompt {prompt_key} added !")

    def insert_conversation(self, prompt_datum: Dict[str, ChatModelResponse], doc_prefix="memory"):
        """
        :param prompt_datum: Model Responses in List
        :param doc_prefix: memory or prompt
        """
        try:
            pipeline = self.redis_conn.pipeline()

            for idx, dict_data in enumerate(prompt_datum.items(), 1):
                key, data = dict_data
                if not data:
                    self.logger.info(f"No data in {key}!")
                    continue

                agent_role = data.role
                data_key = f"{agent_role}:{data.data_id}:{key}"

                llm_answer = "null"

                if agent_role == "student":
                    if data.function_call:
                        llm_answer = data.function_call.arguments
                else:
                    llm_answer = data.content

                prompt_tokens_usage = data.prompt_tokens_usage
                completion_tokens_usage = data.completion_tokens_usage

                document = {
                    "role": agent_role,
                    "answer": llm_answer,
                    "prompt_tokens_usage": prompt_tokens_usage,
                    "completion_tokens_usage": completion_tokens_usage
                }

                self._set_single_memory(data_id=data_key,
                                        document=document,
                                        doc_prefix=doc_prefix,
                                        pipeline=pipeline)
            pipeline.execute()

        except Exception as e:
            self.logger.error(f"Error in insert_conversation : {e}")
            raise

    def retrieve_memory(self, data_id, field_name, doc_prefix="memory"):
        """
        :param field_name: one of role, answer, prompt_tokens_usage, completion_tokens_usage
        """
        data_id_for_search = self._get_memory_id(data_id, doc_prefix)
        try:
            prompt_value = self.redis_conn.hget(data_id_for_search, field_name)
            if prompt_value:
                return prompt_value.decode("utf-8")
            else:
                self.logger.info(f"Prompt {data_id} not found !")

        except Exception as e:
            self.logger.error(f"Error in get_prompt : {e}")
            raise

    def get_keys_by_ids_with_lua(self, data_ids):
        """
        :param data_ids: 데이터베이스에 저장된 데이터가 있는지 확인
        :return: list of keys
        """
        lua_script = """
        local keys = {}
        for _, data_id in ipairs(ARGV) do
            local key = "memory:student:" .. data_id .. ":3"
            if redis.call("exists", key) == 1 then
                table.insert(keys, key)
            end
        end
        return keys
        """

        keys = self.redis_conn.eval(lua_script, 0, *data_ids)
        return [key.decode('utf-8') for key in keys]
