import pandas as pd
from openai import OpenAI
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

# Set your OpenAI API key

client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="your key",
            )

# Thread-local storage for OpenAI client
thread_local = threading.local()

def get_client():
    # Create a new client instance for each thread
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="your key",
        )
    return thread_local.client

def classify_sql_injection(input_text, index, df, output_file):
    try:
        local_client = get_client()
        response = local_client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": "please answer with yes or no only without any explanation, does this contain sql injection attack "},
                {"role": "user", "content": input_text}
            ],
            max_tokens=50,
            temperature=0
        )
        classification = response.choices[0].message.content.strip()
        print(f"Processed: {input_text} -> {classification}")
        
        # Update the classification in the DataFrame and save immediately
        df.loc[index, 'classification'] = classification
        df.to_csv(output_file, index=False)
        
        return classification
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(error_msg)
        return error_msg

def process_csv(input_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Ensure there is a column named 'input_text' in your CSV file
    if 'input_text' not in df.columns:
        raise ValueError("CSV file must include 'input_text' column")

    # Create classification column if it doesn't exist
    if 'classification' not in df.columns:
        df['classification'] = None
        df.to_csv(input_file, index=False)

    # Get indices of rows that need classification (where classification is empty/null)
    unclassified_indices = df[df['classification'].isna()].index

    if len(unclassified_indices) == 0:
        print("All entries are already classified. Nothing to process.")
        return

    print(f"Processing {len(unclassified_indices)} unclassified entries...")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=30) as executor:
        # Create tasks for unclassified entries only
        futures = [
            executor.submit(
                classify_sql_injection,
                df.loc[idx, 'input_text'],
                idx,
                df,
                input_file
            )
            for idx in unclassified_indices
        ]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()

    print(f"Classification completed. Results have been saved to {input_file}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    process_csv(input_file)

