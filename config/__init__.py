from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FOLDER = BASE_DIR / "data"
CVS_FOLDER  = DATA_FOLDER / "samples"

CVS_JUNIOR = CVS_FOLDER / "junior"
CVS_TRAINEE = CVS_FOLDER / "trainee"
CVS_CIRA = CVS_FOLDER / "cira"


VECTOR_STORE_PATH = DATA_FOLDER/"vectorstore/resumes"
VECTOR_DB = DATA_FOLDER/"vectorstore"

VECTOR_STORE_CIRA = VECTOR_DB/"cira"
VECTOR_STORE_JUNIOR = VECTOR_DB/"junior"
VECTOR_STORE_TRAINEE = VECTOR_DB/"trainee"

if __name__ == "__main__":
    print(BASE_DIR)