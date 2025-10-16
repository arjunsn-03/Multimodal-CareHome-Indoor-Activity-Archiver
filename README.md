# Multimodal-CareHome-Archiver

## Project Title: Safety-Critical Vision-to-Text Archival System for Elderly Monitoring

### Overview
This project implements a highly efficient **Vision-Only AI Pipeline** designed to solve the data storage and auditing challenges of continuous video surveillance in care home environments. Instead of storing large video files, the system uses advanced Computer Vision (CV) to filter footage for critical events and leverages Generative AI (LLMs/VLMs) to convert those events into detailed, forensic-style text logs.

This approach achieves **massive data compression** while producing **actionable, time-stamped reports** essential for medical and compliance auditing.

### Key Features

* **Intelligent Filtering:** Only processes frames deemed critical or routine by a fast CV model, ignoring hours of static footage.
* **Critical Event Detection:** Implements custom **YOLOv8-Pose heuristics** to accurately identify patient falls and abnormal horizontal postures.
* **Abstractive Archival:** Uses the **Gemini API** to fuse visual data and logic into a coherent, flowing narrative summary.
* **Structured Output:** Generates a final text log separated into two auditable sections: `## CRITICAL INCIDENT REPORT ##` and `## ROUTINE ACTIVITY LOG ##`.

### Architecture and Workflow

The system operates as a filter-and-reasoning pipeline:

1.  **Detection Layer (Local GPU):** **OpenCV** reads the video. **YOLOv8-Pose** extracts keypoints (joints). Python logic applies heuristics to create `CRITICAL_TIMESTAMPS`.
2.  **Extraction Layer:** Only frames corresponding to the critical and routine timestamps are extracted and saved as JPEGs.
3.  **Reasoning Layer (Gemini VLM/LLM API):** Filtered JPEGs are sent to the Gemini VLM for description. The resulting captions are then fused and structured by the Gemini LLM into the final report format.

### Installation and Setup (Colab Environment)

To run this project, you need to set up the environment and provide your API key.

1.  **Install Dependencies:**
    ```bash
    !pip install -q -U google-genai ultralytics opencv-python pillow
    ```
2.  **API Key Configuration:** Obtain a Gemini API Key and set it in your environment or directly within the main Python file (`video_summarizer_vlm.py`).
    ```python
    GEMINI_API_KEY = "YOUR_VALID_KEY_STRING_HERE" 
    ```
3.  **Model Loading:** The code automatically downloads and loads the pre-trained **YOLOv8n-pose.pt** model.

### Key Governing Heuristic Parameters

The system's sensitivity is controlled by these variables, defined in the Python code:

* **`FALL_THRESHOLD_Y`**: **Governs Detection Sensitivity.** Sets the normalized vertical height threshold (e.g., $0.75$) to trigger a fall if the patient's hip keypoints are close to the floor.
* **`MIN_HEIGHT_RATIO`**: **Governs Posture Confirmation.** Confirms the horizontal posture (ratio $< 1.0$) to filter out false alarms from bending.
* **`CRITICAL_WINDOW_SIZE`**: **Governs Context Padding.** Determines the duration (in seconds) that is sampled ($\pm$ seconds) around a detected fall event to capture the full context.

### Project Output Example (Simulated)

The final synthesized report showcases the value of the fusion model:

```markdown
## CRITICAL INCIDENT REPORT ##

[03:12:05 - 03:12:35]: Patient began abrupt movement, shifting from an upright posture to a critical descent. At 03:12:35, the patient's torso was confirmed to be horizontal, lying fully on the floor next to the bed railing. The VLM verified the final recumbent position and the absence of staff.

## ROUTINE ACTIVITY LOG ##

[00:00:00 - 06:00:00]: Patient remained stationary and horizontal in bed throughout the night. Routine visual checks confirm a static environment. 
[06:00:00 - 09:00:00]: Patient was seated in the armchair near the window, engaging in quiet activity.
