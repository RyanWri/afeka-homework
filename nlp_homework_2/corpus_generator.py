import asyncio
from ollama import AsyncClient


# Async function to generate random sentences using Ollama
async def generate_random_sentence_ollama(model, num_sentences, file_name):
    sentences = []

    async def fetch_sentence():
        prompt = """
            Pick a random topic related to royalty, gender, or fruits, and generate a valid sentence 
            with proper grammar. The sentence should include one of the following words: 
            king, queen, man, woman, apple and orange. 
            The sentence should be no longer than 12 words with no special characters. 
            Ensure that each sentence is unique and the topics are different from each other.
            """
        # Send the message asynchronously
        response = await AsyncClient().chat(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()

    # Use asyncio.gather to fetch all sentences concurrently
    tasks = [fetch_sentence() for _ in range(num_sentences)]
    sentences = await asyncio.gather(*tasks)

    # Write sentences to a file
    with open(file_name, "a") as file:
        file.write("\n".join(sentences) + "\n")

    print(f"{num_sentences} random sentences generated and saved to {file_name}")


# Run the async function using asyncio
if __name__ == "__main__":
    asyncio.run(
        generate_random_sentence_ollama(
            model="mistral", num_sentences=30, file_name="data/custom_corpus.txt"
        )
    )
