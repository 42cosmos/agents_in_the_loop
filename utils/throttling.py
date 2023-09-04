import logging
import time
from threading import Lock


class Throttling:
    # Token Bucket 알고리즘을 사용하여 요청 속도를 제한한다
    def __init__(self, rate):
        # Thread-safety를 위해 Lock을 사용
        self._consume_lock = Lock()
        # 미리 정해진 숫자만큼의 토큰을 가진 버킷
        self.rate = rate
        # init no tokens
        self.tokens = 0
        # last token request time
        self.timestamp = None

    def consume_with_tokens(self, tokens=0, amount=1):
        with self._consume_lock:
            # 호출 시간
            now = time.time()

            # 시간 측정은 첫 번째 토근에 대해 초기화함으로써 초기 부하를 줄임
            if self.timestamp is None:  # 최초 실행 시
                self.timestamp = now

            # 경과 시간을 계산
            elapsed = now - self.timestamp

            # 초과된 시간이 새로운 토큰을 추가할 만큼 충분히 긴지 확인한다
            if elapsed * self.rate > 1:
                # 새 토큰 추가
                self.tokens += elapsed * self.rate
                self.timestamp = now

            # 토큰의 최대 개수를 초과하지 않도록 한다
            self.tokens = min(self.rate, self.tokens)

            # 토큰이 충분하지 않으면 0을 반환하고, 그렇지 않으면 토큰을 소비하고 소비한 토큰의 개수를 반환한다
            if self.tokens < amount:
                return 0
            else:
                self.tokens -= amount  # Deduct tokens
                return amount


class TokenThrottling(Throttling):
    def __init__(self, rate, token_rate):
        super().__init__(rate)
        # rate: 초당 API 호출 횟수 제한
        # token rate: 초당 토큰 자용량 제한
        self.token_rate = token_rate
        self.token_bucket = 0
        self.token_last = None
        self.logger = logging.getLogger(f"{TokenThrottling.__name__}")
        self.logger.setLevel(logging.INFO)

    def consume_with_tokens(self, tokens, amount=1):
        with self._consume_lock:
            now = time.time()

            if self.timestamp is None:  # 최초 실행
                self.timestamp = now

            if self.token_last is None:
                self.token_last = now

            elapsed = now - self.timestamp
            token_elapsed = now - self.token_last

            if elapsed * self.rate > 1:
                # 새 토큰 추가
                self.tokens += elapsed * self.rate
                self.timestamp = now

            if token_elapsed * self.token_rate > 1:
                self.token_bucket += token_elapsed * self.token_rate
                self.token_last = now

            # 토큰의 최대 개수를 초과하지 않도록 보장
            self.tokens = min(self.rate, self.tokens)
            self.token_bucket = min(self.token_rate, self.token_bucket)

            # 토큰이 충분해질 때까지 대기
            while self.tokens < amount or self.token_bucket < tokens:
                time.sleep(60)  # 60초 대기
                self.logger.info(f"Waiting for tokens to be refilled...")
                now = time.time()
                elapsed = now - self.timestamp
                token_elapsed = now - self.token_last

                if elapsed * self.rate > 1:
                    self.tokens += elapsed * self.rate
                    self.timestamp = now

                if token_elapsed * self.token_rate > 1:
                    self.token_bucket += token_elapsed * self.token_rate
                    self.token_last = now

                self.tokens = min(self.rate, self.tokens)
                self.token_bucket = min(self.token_rate, self.token_bucket)

            self.tokens -= amount
            self.token_bucket -= tokens
            return amount