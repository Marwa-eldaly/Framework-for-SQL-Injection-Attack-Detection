import pandas as pd
import openai
import sys

# Set your OpenAI API key
openai.api_key = 'your key'

def classify_sql_injection(input_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Ensure you use a model that is accessible to you
            messages=[
                {"role": "system", "content": "please answer with yes or no only without any explanation, does this contain sql injection attack "},
                {"role": "user", "content": input_text}
            ],
            max_tokens=50,
            temperature=0
        )
        classification = response['choices'][0]['message']['content'].strip()
        return classification
    except openai.error.RateLimitError:
        return "Rate limit exceeded. Please try again later."
    except openai.error.InvalidRequestError as e:
        return f"An error occurred: {str(e)}"

def process_csv(input_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Ensure there is a column named 'input_text' in your CSV file
    if 'input_text' not in df.columns:
        raise ValueError("CSV file must include 'input_text' column")

    # Classify each row and save the result in a new column
    df['classification'] = df['input_text'].apply(classify_sql_injection)

    # Save the modified DataFrame back to the same CSV file
    df.to_csv(input_file, index=False)
    print(f"Classification results have been added to {input_file}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    process_csv(input_file)

