import redis

class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host=host, port=port, db=db)

    def set_value(self, key, value):
        self.r.set(key, value)

    def get_value(self, key):
        value = self.r.get(key)
        if value:
            return value.decode('utf-8')
        return None

    def get_all_keys(self):
        """
        Get all keys from redis
        Beware of using this function in production...
        :return:
        """
        return [key.decode('utf-8') for key in self.r.keys()]

    def add_to_set(self, key, *values):
        self.r.sadd(key, *values)

    def get_all_from_set(self, key):
        return [value.decode('utf-8') for value in self.r.smembers(key)]

    def push_to_list(self, key, *values):
        self.r.lpush(key, *values)

    def get_all_from_list(self, key):
        return [value.decode('utf-8') for value in self.r.lrange(key, 0, -1)]
