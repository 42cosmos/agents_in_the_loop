import yaml
from easydict import EasyDict
from utils import Student


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    student = Student(config=config)
    question = "Solve the equation x^2 - 3x - 4 = 0."
    answer = student.talk_to_agent(question)
