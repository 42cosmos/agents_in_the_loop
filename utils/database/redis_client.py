import redis


class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host=host, port=port, db=db)

    def get_all_keys(self, pattern='*'):
        return [key.decode('utf-8') for key in self.r.keys(pattern)]

    def get_values_by_key_pattern(self, pattern):
        keys = self.get_all_keys(pattern)
        return {key: self.r.get(key).decode('utf-8') if self.r.get(key) else None for key in keys}

    def set_value(self, key, value):
        self.r.set(key, value)

    def get_value(self, key):
        value = self.r.get(key)
        if value:
            return value.decode('utf-8')
        return None

    def add_to_set(self, key, *values):
        self.r.sadd(key, *values)

    def get_all_from_set(self, key):
        return [value.decode('utf-8') for value in self.r.smembers(key)]

    def push_to_list(self, key, *values):
        self.r.lpush(key, *values)

    def get_all_from_list(self, key):
        return [value.decode('utf-8') for value in self.r.lrange(key, 0, -1)]

    def delete_key(self, key):
        return self.r.delete(key)