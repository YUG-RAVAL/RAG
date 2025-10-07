# dataset_manager.py
import os
from opik import Opik
from dotenv import load_dotenv

load_dotenv()
OPIK_API_KEY = os.getenv("OPIK_API_KEY")

def create_rag_dataset(dataset_name="Samsung-LCD-RAG-Questions"):
    """Create and populate a dataset with RAG test questions."""
    client = Opik(api_key=OPIK_API_KEY)
    
    # Get or create the dataset
    dataset = client.get_or_create_dataset(name=dataset_name)
    
    # Example questions - replace with your actual questions
    questions = [
  {
    "input": "What should be done before using the LCD monitor?",
    "expected_output": "Read all safety precautions and refer to the Troubleshooting section if a problem occurs."
  },
  {
    "input": "What is the purpose of keeping a 10 cm distance from the wall when installing the monitor?",
    "expected_output": "To ensure proper ventilation and prevent internal overheating that could cause fire."
  },
  {
    "input": "How should you clean the monitor screen?",
    "expected_output": "Use a soft, dry cloth. Do not spray water or use chemicals like benzene or thinner."
  },
  {
    "input": "What does the 'MagicBright' feature do?",
    "expected_output": "It optimizes the viewing environment with modes like Entertain, Internet, Text, Dynamic Contrast, and Custom."
  },
  {
    "input": "What happens if a static image is displayed for too long?",
    "expected_output": "It may create a persistent image or stain on the screen."
  },
  {
    "input": "How can the monitor be connected to a computer with analog output?",
    "expected_output": "Use a D-Sub cable to connect the PC's D-Sub port to the monitor's PC IN port."
  },
  {
    "input": "What does the Plug & Play feature do?",
    "expected_output": "It enables automatic setup when the TV is powered on for the first time."
  },
  {
    "input": "What function does the Kensington Lock provide?",
    "expected_output": "It secures the monitor against theft, especially in public areas."
  },
  {
    "input": "What should you do if the monitor shows a blank screen?",
    "expected_output": "Run the self-diagnosis: disconnect the PC cable and check if 'Check Signal Cable' appears."
  },
  {
    "input": "How can a user optimize power usage on the monitor?",
    "expected_output": "Use the Energy Saving mode set to Low, Medium, High, or Auto to reduce power consumption."
  }
]


    
    # Insert questions into the dataset
    dataset.insert(questions)
    
    print(f"Dataset '{dataset_name}' created with {len(questions)} questions")
    return dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage RAG evaluation datasets")
    parser.add_argument("--create", action="store_true", help="Create a new dataset", default=True)
    parser.add_argument("--name", default="Samsung-LCD-RAG-Questions", help="Dataset name")
    args = parser.parse_args()
    
    if args.create:
        create_rag_dataset(args.name)
    else:
        # Display existing datasets
        client = Opik(api_key=OPIK_API_KEY)
        dataset = client.get_dataset(name=args.name)
        df = dataset.to_pandas()
        print(f"Dataset '{args.name}' has {len(df)} questions")
        print(df.head())