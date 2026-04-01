from pathlib import Path
from Angiogram_DICOM_KeyFrame_Extraction import process_angiogram

# ----------------------------------------
# CONFIGURE TEST FILE HERE
# ----------------------------------------
TEST_FILE = Path("C:/Users/User/Documents/IIT Stage 2/IIT Stage 2 Semester 1/CM2603  Data Science Group Project/Angiogram Videos/DICOM/Patient-0002")
# OR
# TEST_FILE = Path("test_data/sample_video.mp4")


def main():
    if not TEST_FILE.exists():
        print("Test file not found:", TEST_FILE)
        return

    print("Testing angiogram pipeline...")
    print("Input file:", TEST_FILE)

    try:
        result = process_angiogram(str(TEST_FILE))

        print("\n--- PROCESS SUCCESSFUL ---")
        print("Patient ID:", result["patient_id"])
        print("Output Directory:", result["output_directory"])
        print("Selected Frames:", result["selected_frame_indices"])

    except Exception as e:
        print("\n--- PROCESS FAILED ---")
        print("Error:", str(e))


if __name__ == "__main__":
    main()