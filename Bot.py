import transformers
import speech_recognition as sr

class ConversationalBot:
    def __init__(self, model_name="DialoGPT-large"):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        # List to store the user's responses.
        self.user_responses = []

    def train(self, dataset):
        """Trains the model on the given dataset.

        Args:
            dataset: A list of (input_text, output_text) pairs.
        """

        inputs = []
        labels = []
        for input_text, output_text in dataset:
            inputs.append(self.tokenizer(input_text, return_tensors="pt").input_ids)
            labels.append(self.tokenizer(output_text, return_tensors="pt").input_ids)

        self.model.train()
        loss = self.model(inputs, labels=labels).loss
        loss.backward()
        self.model.optimizer.step()

    def generate_response(self, input_text):
        """Generates a response to the given input text.

        Args:
            input_text: The input text to generate a response for.

        Returns:
            A string containing the generated response.
        """

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids, max_length=1024)
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

    def train_on_command(self, command):
        """Trains the model on the given command.

        Args:
            command: The command to train the model on.
        """

        if command == "train yourself":
            # Train the model on the list of user responses.
            self.train(self.user_responses)

    def take_input(self):
        """Takes input from the user as text and voice.

        Returns:
            A string containing the user's input.
        """

        # Take text input from the user.
        text_input = input("Enter your input: ")

        # Take voice input from the user.
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio_input = r.listen(source)

        try:
            voice_input = r.recognize_google(audio_input)
        except sr.UnknownValueError:
            voice_input = None

        # Return the user's input.
        return text_input or voice_input

    def main(self):
        while True:
            # Get the user's input.
            user_input = self.take_input()

            # Add the user's response to the list.
            self.user_responses.append(user_input)

            # If the user input is "train yourself", train the model.
            if user_input == "train yourself":
                self.train_on_command(user_input)
            else:
                # Generate a response to the user's input.
                response = self.generate_response(user_input)

                # Print the response.
                print(response)

if __name__ == "__main__":
    bot = ConversationalBot()
    bot.main()
