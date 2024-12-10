# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "pillow",
#   "python-dotenv",
#   "requests",
# ]
# ///

from dotenv import load_dotenv
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Load .env file
load_dotenv()

# Get API key from .env
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set in the .env file.")
    sys.exit(1)

# AI Proxy URL
AI_PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"


def analyze_dataset(file_path):
    """
    Analyze the dataset and return summaries, statistics, and insights.
    
    Args:
        file_path (str): Path to the CSV file to analyze.
    
    Returns:
        df (DataFrame): The dataset in a pandas DataFrame.
        summary (dict): A dictionary containing basic information about the dataset.
        numeric_summary (dict): A statistical summary of numeric columns.
        correlation_matrix (DataFrame): Correlation matrix of numeric columns.
    """
    try:
        # Load dataset with explicit encoding
        df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Try this common encoding first

        # Basic information
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "example_values": df.head(3).to_dict(),
        }

        # Statistical summary for numeric columns
        numeric_summary = df.describe().to_dict()

        # Correlation matrix (for numeric data)
        correlation_matrix = df.corr(numeric_only=True)

        return df, summary, numeric_summary, correlation_matrix

    except UnicodeDecodeError as e:
        print(f"Error decoding the dataset: {e}")
        print("Try using a different encoding, e.g., 'ISO-8859-1' or 'latin1'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or analyzing the dataset: {e}")
        sys.exit(1)


def visualize_dataset(df, correlation_matrix, file_path):
    """
    Generate visualizations specific to the dataset and save as PNG files.
    
    Args:
        df (DataFrame): The dataset to generate visualizations for.
        correlation_matrix (DataFrame): The correlation matrix for the dataset.
        file_path (str): The path to the dataset file.
    
    Returns:
        charts (list): A list of generated chart file names.
    """
    charts = []
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    # File-specific visualizations
    if "goodreads" in file_path.lower():
        # Histogram of Average Ratings
        if "average_rating" in df.columns:
            plt.figure(figsize=(8, 6))
            df["average_rating"].hist(bins=30, color="purple", edgecolor="black")
            plt.title("Distribution of Goodreads Ratings")
            plt.xlabel("Average Rating")
            plt.ylabel("Frequency")
            plt.savefig("goodreads_ratings.png")
            charts.append("goodreads_ratings.png")
            plt.close()
        else:
            print("Column 'average_rating' is missing. Skipping histogram.")

        # Bar Chart: Top 10 Authors by Ratings Count
        if "authors" in df.columns and "ratings_count" in df.columns:
            plt.figure(figsize=(10, 6))
            top_authors = df.groupby("authors")["ratings_count"].sum().nlargest(10)
            top_authors.plot(kind="bar", color="lightblue", edgecolor="black")
            plt.title("Top 10 Authors by Total Ratings Count")
            plt.xlabel("Authors")
            plt.ylabel("Total Ratings Count")
            plt.tight_layout()
            plt.savefig("goodreads_top_authors.png")
            charts.append("goodreads_top_authors.png")
            plt.close()
        else:
            print("Required columns for bar chart are missing. Skipping top authors chart.")

        # Scatter Plot: Average Rating vs. Ratings Count
        if "average_rating" in df.columns and "ratings_count" in df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x="average_rating", y="ratings_count", alpha=0.6)
            plt.title("Average Rating vs Ratings Count")
            plt.xlabel("Average Rating")
            plt.ylabel("Ratings Count")
            plt.tight_layout()
            plt.savefig("goodreads_ratings_vs_count.png")
            charts.append("goodreads_ratings_vs_count.png")
            plt.close()
        else:
            print("Required columns for scatter plot are missing. Skipping ratings vs count.")

        # Line Plot: Average Rating Over Years
        if "original_publication_year" in df.columns and "average_rating" in df.columns:
            plt.figure(figsize=(12, 6))
            avg_rating_by_year = df.groupby("original_publication_year")["average_rating"].mean()
            avg_rating_by_year.plot(kind="line", color="orange", linewidth=2)
            plt.title("Average Rating Over Years")
            plt.xlabel("Original Publication Year")
            plt.ylabel("Average Rating")
            plt.grid()
            plt.tight_layout()
            plt.savefig("goodreads_rating_over_years.png")
            charts.append("goodreads_rating_over_years.png")
            plt.close()
        else:
            print("Required columns for line plot are missing. Skipping rating over years.")

        # Correlation Heatmap as Fallback
        if len(charts) < 2:
            if not correlation_matrix.empty:
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
                plt.title("Correlation Heatmap")
                heatmap_name = f"goodreads_correlation_heatmap.png"
                plt.savefig(heatmap_name)
                charts.append(heatmap_name)
                plt.close()
            else:
                print("Correlation matrix is empty. Skipping heatmap.")

    elif "happiness" in file_path.lower():
        # Bar Chart: Average Happiness by Country
        if "Country name" in df.columns and "Life Ladder" in df.columns:
            plt.figure(figsize=(12, 8))
            avg_happiness = df.groupby("Country name")["Life Ladder"].mean().sort_values()
            avg_happiness.plot(kind="barh", color="skyblue", edgecolor="black")
            plt.title("Average Happiness Score by Country")
            plt.xlabel("Life Ladder (Average Happiness Score)")
            plt.ylabel("Country")
            plt.tight_layout()
            plt.savefig("average_happiness_by_country.png")
            charts.append("average_happiness_by_country.png")
            plt.close()

        # Line Plot: Happiness Over Time by Country
        if "year" in df.columns and "Life Ladder" in df.columns:
            plt.figure(figsize=(12, 8))
            for country in df["Country name"].unique():
                country_data = df[df["Country name"] == country]
                plt.plot(country_data["year"], country_data["Life Ladder"], label=country, alpha=0.5)

            plt.title("Happiness Over Time by Country")
            plt.xlabel("Year")
            plt.ylabel("Life Ladder (Happiness Score)")
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize="small")
            plt.tight_layout()
            plt.savefig("happiness_over_time.png")
            charts.append("happiness_over_time.png")
            plt.close()

        # Scatter Plot: GDP vs Happiness
        if "Log GDP per capita" in df.columns and "Life Ladder" in df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x="Log GDP per capita", y="Life Ladder", hue="Country name", legend=False, alpha=0.6)
            plt.title("GDP vs Happiness")
            plt.xlabel("Log GDP per capita")
            plt.ylabel("Life Ladder (Happiness Score)")
            plt.tight_layout()
            plt.savefig("gdp_vs_happiness.png")
            charts.append("gdp_vs_happiness.png")
            plt.close()

        # Correlation Heatmap
        if not correlation_matrix.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            heatmap_name = f"happiness_correlation_heatmap.png"
            plt.savefig(heatmap_name)
            charts.append(heatmap_name)
            plt.close()

    elif "media" in file_path.lower():
        # Histogram of Ratings
        if "overall" in df.columns:
            plt.figure(figsize=(8, 6))
            df["overall"].hist(bins=30, color="green", edgecolor="black")
            plt.title("Distribution of Media Overall Ratings")
            plt.xlabel("Overall Rating")
            plt.ylabel("Frequency")
            plt.savefig("media_ratings_distribution.png")
            charts.append("media_ratings_distribution.png")
            plt.close()

        # Box Plot: Ratings by Media Type
        if "type" in df.columns and "overall" in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="type", y="overall", hue="type", dodge=False, palette="pastel")
            plt.title("Overall Ratings by Media Type")
            plt.xlabel("Media Type")
            plt.ylabel("Overall Rating")
            plt.savefig("media_ratings_by_type.png")
            charts.append("media_ratings_by_type.png")
            plt.close()

        # Scatter Plot: Quality vs Repeatability
        if "quality" in df.columns and "repeatability" in df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x="quality", y="repeatability", alpha=0.6)
            plt.title("Quality vs Repeatability")
            plt.xlabel("Quality Rating")
            plt.ylabel("Repeatability Score")
            plt.tight_layout()
            plt.savefig("quality_vs_repeatability.png")
            charts.append("quality_vs_repeatability.png")
            plt.close()

        # Bar Chart: Average Overall Ratings by Language
        if "language" in df.columns and "overall" in df.columns:
            plt.figure(figsize=(10, 6))
            avg_ratings_by_language = df.groupby("language")["overall"].mean().sort_values()
            avg_ratings_by_language.plot(kind="bar", color="skyblue", edgecolor="black")
            plt.title("Average Overall Ratings by Language")
            plt.xlabel("Language")
            plt.ylabel("Average Overall Rating")
            plt.tight_layout()
            plt.savefig("media_avg_ratings_by_language.png")
            charts.append("media_avg_ratings_by_language.png")
            plt.close()

        # Correlation Heatmap
        if not correlation_matrix.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            heatmap_name = f"media_correlation_heatmap.png"
            plt.savefig(heatmap_name)
            charts.append(heatmap_name)
            plt.close()

    return charts


def generate_story(summary, numeric_summary, charts, file_path):
    """
    Use the AI Proxy (GPT-4o-Mini) to generate a story about the analysis.
    
    Args:
        summary (dict): Summary of the dataset.
        numeric_summary (dict): Statistical summary of numeric columns.
        charts (list): List of visualizations created.
        file_path (str): Path to the dataset file.
    
    Returns:
        story (str): A detailed analysis report.
    """
    dataset_type = "Goodreads dataset" if "goodreads" in file_path.lower() else (
        "Happiness dataset" if "happiness" in file_path.lower() else "Media dataset"
    )

    prompt = f"""
    I analyzed a {dataset_type} with the following characteristics:
    - Shape: {summary['shape']}
    - Columns: {summary['columns']}
    - Data Types: {summary['dtypes']}
    - Missing Values: {summary['missing_values']}
    - Example Values: {summary['example_values']}

    I performed basic statistics and found the following:
    {numeric_summary}

    I created the following visualizations:
    - {', '.join(charts)}

    Write a detailed story about the dataset, the analysis performed, the insights discovered, and the implications of the findings.
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a data analysis expert."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 800,
            "temperature": 0.7,
        }
        response = requests.post(AI_PROXY_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating story: {e}")
        sys.exit(1)


def save_readme(story, charts, dataset_name):
    """
    Save the story and charts into a dataset-specific README file.
    
    Args:
        story (str): The generated story from the AI.
        charts (list): List of chart file names.
        dataset_name (str): Name of the dataset.
    """
    readme_filename = f"README_{dataset_name}.md"
    with open(readme_filename, "w") as f:
        f.write("# Analysis Report\n\n")
        f.write(story)
        f.write("\n\n## Visualizations\n")
        for chart in charts:
            f.write(f"![{chart}]({chart})\n")
    print(f"Results saved to {readme_filename}")

if __name__ == "__main__":
    # Check for CSV filename
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Running analysis for: {file_path}")

    # Step 1: Analyze dataset
    print("Step 1: Analyzing dataset...")
    df, summary, numeric_summary, correlation_matrix = analyze_dataset(file_path)
    print("Dataset analyzed successfully.")

    # Step 2: Visualize dataset
    print("Step 2: Generating visualizations...")
    charts = visualize_dataset(df, correlation_matrix, file_path)
    print(f"Visualizations created: {charts}")

    # Step 3: Generate story
    print("Step 3: Generating story using AI Proxy...")
    story = generate_story(summary, numeric_summary, charts, file_path)
    print("Story generated successfully.")

    # Step 4: Save results to README
    print(f"Step 4: Saving results to README_{dataset_name}.md...")
    save_readme(story, charts, dataset_name)
    print(f"Analysis complete for {file_path}. Results saved to README_{dataset_name}.md and PNG files.")
