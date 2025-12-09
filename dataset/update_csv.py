import pandas as pd

def main():
    # Load the CSV file
    df = pd.read_csv("layout_info.csv")
    
    # Drop the unwanted columns
    df = df.drop(columns=["short_id", "dataset"])
    
    # Save the cleaned CSV
    df.to_csv("layout_info_cleaned.csv", index=False)

if __name__ == "__main__":
    main()