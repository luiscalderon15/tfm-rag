from pathlib import Path
import re
import time

import pandas as pd
import matplotlib.pyplot as plt

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def count_words(text: str) -> int:
    """Count the number of words in a text."""
    if not isinstance(text, str) or not text.strip():
        return 0

    return len(re.findall(r"\b\w+\b", text))


# ============================================================
# PYMUPDF
# ============================================================

def extract_cv_pymupdf(file_path: Path) -> Document:
    """
    Extract text from a PDF using PyMuPDF and measure
    the extraction time.
    """

    start = time.perf_counter()

    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    elapsed = time.perf_counter() - start

    text = "\n".join(page.page_content for page in pages)

    metadata = {
        "file_name": file_path.name,
        "method": "PyMuPDF",
        "n_pages": len(pages),
        "n_char": len(text),
        "n_words": count_words(text),
        "extraction_time": elapsed,
    }

    return Document(
        page_content=text,
        metadata=metadata,
    )


# ============================================================
# DOCLING
# ============================================================

def extract_cv_docling(
    file_path: Path,
    converter: DocumentConverter,
) -> Document:
    """
    Extract text from a PDF using Docling and measure
    the extraction time.
    """

    start = time.perf_counter()

    result = converter.convert(str(file_path))

    elapsed = time.perf_counter() - start

    text = result.document.export_to_markdown()

    metadata = {
        "file_name": file_path.name,
        "method": "Docling",
        "n_pages": None,
        "n_char": len(text),
        "n_words": count_words(text),
        "extraction_time": elapsed,
    }

    return Document(
        page_content=text,
        metadata=metadata,
    )


# ============================================================
# PROCESS FOLDER
# ============================================================

def process_folder(folder_path: Path) -> pd.DataFrame:
    """
    Process all PDFs in a folder using PyMuPDF and Docling.
    """

    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in: {folder_path}"
        )

    print(f"Found {len(pdf_files)} PDF files.")

    results = []

    # Create Docling converter only once
    converter = DocumentConverter()

    for i, pdf_file in enumerate(pdf_files, start=1):

        print(
            f"[{i}/{len(pdf_files)}] "
            f"Processing {pdf_file.name}"
        )

        # -------------------------
        # PyMuPDF
        # -------------------------

        doc_pymupdf = extract_cv_pymupdf(pdf_file)

        results.append(doc_pymupdf.metadata)

        # -------------------------
        # Docling
        # -------------------------

        doc_docling = extract_cv_docling(
            pdf_file,
            converter
        )

        results.append(doc_docling.metadata)

    return pd.DataFrame(results)


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each extraction method.
    """

    statistics = (
        df
        .groupby("method")["extraction_time"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
    )

    return statistics


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def monte_carlo_simulation(
    df: pd.DataFrame,
    n_simulations: int = 10000,
) -> pd.DataFrame:
    """
    Monte Carlo simulation based on the empirical distribution
    of the measured extraction times.

    Each simulation samples, with replacement, from the
    real extraction times observed in the experiment.
    """

    simulations = []

    for method in df["method"].unique():

        times = (
            df[df["method"] == method]
            ["extraction_time"]
            .values
        )

        # Generate simulations
        simulated_times = []

        for _ in range(n_simulations):

            sample = pd.Series(times).sample(
                n=len(times),
                replace=True,
            )

            simulated_times.append(sample.mean())

        for value in simulated_times:

            simulations.append({
                "method": method,
                "simulated_mean_time": value,
            })

    return pd.DataFrame(simulations)


# ============================================================
# HISTOGRAM
# ============================================================

def plot_histograms(
    df: pd.DataFrame,
    output_folder: Path,
):
    """
    Plot the distribution of real extraction times.
    """

    for method in df["method"].unique():

        data = df[
            df["method"] == method
        ]["extraction_time"]

        plt.figure(figsize=(8, 5))

        plt.hist(
            data,
            bins=30,
        )

        plt.xlabel("Extraction time (seconds)")
        plt.ylabel("Number of CVs")
        plt.title(
            f"Extraction time distribution - {method}"
        )

        plt.tight_layout()

        plt.savefig(
            output_folder
            / f"histogram_{method}.png",
            dpi=300,
        )

        plt.show()


# ============================================================
# MONTE CARLO HISTOGRAM
# ============================================================

def plot_monte_carlo(
    simulation_df: pd.DataFrame,
    output_folder: Path,
):
    """
    Plot the distribution of Monte Carlo simulated
    mean extraction times.
    """

    for method in simulation_df["method"].unique():

        data = simulation_df[
            simulation_df["method"] == method
        ]["simulated_mean_time"]

        plt.figure(figsize=(8, 5))

        plt.hist(
            data,
            bins=30,
        )

        plt.xlabel(
            "Simulated mean extraction time (seconds)"
        )

        plt.ylabel("Number of simulations")

        plt.title(
            f"Monte Carlo simulation - {method}"
        )

        plt.tight_layout()

        plt.savefig(
            output_folder
            / f"monte_carlo_{method}.png",
            dpi=300,
        )

        plt.show()


# ============================================================
# SCALABILITY
# ============================================================

def plot_scalability(
    df: pd.DataFrame,
    output_folder: Path,
):
    """
    Show the estimated total processing time as the number
    of CVs increases.
    """

    n_cvs = [
        1,
        10,
        50,
        100,
        500,
        1000,
    ]

    plt.figure(figsize=(8, 5))

    for method in df["method"].unique():

        mean_time = (
            df[df["method"] == method]
            ["extraction_time"]
            .mean()
        )

        total_times = [
            n * mean_time
            for n in n_cvs
        ]

        plt.plot(
            n_cvs,
            total_times,
            marker="o",
            label=method,
        )

    plt.xlabel("Number of CVs")
    plt.ylabel("Estimated total time (seconds)")
    plt.title("Estimated processing time vs number of CVs")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_folder / "scalability.png",
        dpi=300,
    )

    plt.show()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(folder_path: str):

    folder = Path(folder_path)

    # Create output folder
    output_folder = folder / "benchmark_results"
    output_folder.mkdir(
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("PDF EXTRACTION BENCHMARK")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. PROCESS PDFs
    # --------------------------------------------------------

    df = process_folder(folder)

    # --------------------------------------------------------
    # 2. SAVE RAW RESULTS
    # --------------------------------------------------------

    df.to_csv(
        output_folder / "extraction_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. STATISTICS
    # --------------------------------------------------------

    statistics = calculate_statistics(df)

    print("\n")
    print("=" * 60)
    print("STATISTICS")
    print("=" * 60)

    print(statistics)

    statistics.to_csv(
        output_folder / "statistics.csv"
    )

    # --------------------------------------------------------
    # 4. SPEEDUP
    # --------------------------------------------------------

    mean_pymupdf = (
        df[df["method"] == "PyMuPDF"]
        ["extraction_time"]
        .mean()
    )

    mean_docling = (
        df[df["method"] == "Docling"]
        ["extraction_time"]
        .mean()
    )

    speedup = mean_docling / mean_pymupdf

    print("\n")
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(
        f"PyMuPDF mean: {mean_pymupdf:.4f} s"
    )

    print(
        f"Docling mean: {mean_docling:.4f} s"
    )

    print(
        f"Docling is approximately "
        f"{speedup:.1f}x slower than PyMuPDF"
    )

    # --------------------------------------------------------
    # 5. MONTE CARLO
    # --------------------------------------------------------

    simulation_df = monte_carlo_simulation(
        df,
        n_simulations=10000,
    )

    simulation_df.to_csv(
        output_folder / "monte_carlo_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. PLOTS
    # --------------------------------------------------------

    plot_histograms(
        df,
        output_folder,
    )

    plot_monte_carlo(
        simulation_df,
        output_folder,
    )

    plot_scalability(
        df,
        output_folder,
    )

    print("\n")
    print("=" * 60)
    print("EXPERIMENT FINISHED")
    print("=" * 60)

    print(
        f"\nResults saved to:\n"
        f"{output_folder}"
    )

BASE_DIR = Path(__file__).resolve().parents[1]

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    folder_path = BASE_DIR/"data/curated"

    run_experiment(folder_path)