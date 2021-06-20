import ab_models
import ab_tokenizers
import pickle


class AnswerBot:
    """Simple class for AI with text I/O."""
    def __init__(self):
        self.model = ab_models.NumberAI()
        self.tokenizer = ab_tokenizers.TokenNumberizer()
    
    def train_model(self, questions, answers, epochs=1):
        int_questions = [self.tokenizer.stoi(question) for question in questions]
        int_answers = [self.tokenizer.stoi(answer) for answer in answers]
        self.model.train(int_questions, int_answers, epochs=epochs)
    
    def train(self, questions, answers, affected_parts=['model', 'tokenizer'], epochs=1):
        if 'tokenizer' in [string.lower() for string in affected_parts]:
            self.tokenizer.adapt(questions + answers)
        if 'model' in [string.lower() for string in affected_parts]:
            self.train_model(questions, answers, epochs=epochs)
    
    def answer(self, question):
        int_question = self.tokenizer.stoi(question)
        answer = self.tokenizer.itos(int(self.model.predict(int_question)))
        return answer
    
    def save_to_file(self, filename: str):
        with open(filename, 'wb') as botfile:
            pickle.dump(self, botfile)
    
    @staticmethod
    def load_from_file(filename: str):
        with open(filename, 'rb') as botfile:
            return pickle.load(botfile)